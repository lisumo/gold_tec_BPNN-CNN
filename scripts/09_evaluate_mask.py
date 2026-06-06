import sys
import os
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.utils.common import setup_gpu
from src.models.fusion import TECFusionCNNModel


# 简单的 Dataset 类
class EvaluatorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_region_str(region_str):
    if not region_str: return None
    try:
        vals = list(map(float, region_str.split(',')))
        return {'lon_min': vals[0], 'lon_max': vals[1], 'lat_min': vals[2], 'lat_max': vals[3]}
    except:
        return None


def main():
    # 1. 独立的参数解析 (不再依赖 src.utils.options)
    parser = argparse.ArgumentParser(description="Pure RMSE Calculator")
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--mask_region', type=str, default=None, help="LonMin,LonMax,LatMin,LatMax")
    parser.add_argument('--dataset_path', type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型
    log_dir = os.path.join(project_root, 'logs', args.exp_name)
    model_path = os.path.join(log_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # 兼容加载 (weights_only)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint['config']

    # 3. 准备区域掩码 (核心逻辑)
    target_region = parse_region_str(args.mask_region)
    if target_region:
        print(f"🎯 Target Region: {target_region}")
    else:
        print("⚠️ No mask region provided! Calculating GLOBAL RMSE.")

    # 4. 初始化模型
    model = TECFusionCNNModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 获取 Scaler 用于反归一化
    scaler_y = checkpoint['scaler_y']
    scaler_X_spatial = checkpoint.get('scaler_X_spatial') or checkpoint['scaler_X_spatial']
    scaler_X_time = checkpoint.get('scaler_X_time') or checkpoint['scaler_X_time']

    # 5. 加载测试集
    # 默认只跑 Test 集 (因为评估通常只看 Test)
    base_path = args.dataset_path if args.dataset_path else Config.DATASET_PATH
    if '_test.pth' not in base_path:
        base_path = base_path.replace('.pth', '_test.pth')

    if not os.path.exists(base_path):
        # 尝试回退到默认路径
        base_path = os.path.join(project_root, 'dataset', 'tec_2d_enhanced_dataset_test.pth')

    print(f"📂 Loading Test Data: {base_path}")
    try:
        raw_samples = torch.load(base_path, weights_only=False)
    except TypeError:
        raw_samples = torch.load(base_path)

    # 6. 数据预处理 (复用 05 的逻辑，至关重要!)
    processed_samples = []
    for s in raw_samples:
        sp = s[0].numpy()
        C, H, W = sp.shape
        sp_flat = sp.reshape(1, -1)
        sp_norm = scaler_X_spatial.transform(sp_flat).reshape(C, H, W)
        tm = s[1].numpy().reshape(1, -1)
        tm_norm = scaler_X_time.transform(tm).reshape(-1)
        # s[2] 是 target, s[3] 是 mit_mask, s[5] 是 gold_mask (根据 EvaluatorDataset)
        new_sample = [torch.FloatTensor(sp_norm), torch.FloatTensor(tm_norm), *s[2:]]
        processed_samples.append(tuple(new_sample))

    loader = DataLoader(EvaluatorDataset(processed_samples), batch_size=32, shuffle=False)

    # 7. 准备网格 (用于生成区域掩码)
    # 使用 Config 中的经纬度范围
    # 注意：这里我们使用 Config 的 MASK_LAT/LON_MIN/MAX，因为 05 是这么用的
    lat_min = getattr(Config, 'MASK_LAT_MIN', -40)
    lat_max = getattr(Config, 'MASK_LAT_MAX', 40)
    lon_min = getattr(Config, 'MASK_LON_MIN', -90)
    lon_max = getattr(Config, 'MASK_LON_MAX', -20)

    H, W = model_config['input_shape']
    lats = np.linspace(lat_min, lat_max, H)
    lons = np.linspace(lon_min, lon_max, W)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 生成静态的区域掩码 (Boolean)
    if target_region:
        static_region_mask = (lon_grid >= target_region['lon_min']) & \
                             (lon_grid <= target_region['lon_max']) & \
                             (lat_grid >= target_region['lat_min']) & \
                             (lat_grid <= target_region['lat_max'])
    else:
        static_region_mask = np.ones_like(lon_grid, dtype=bool)

    # 8. 评估循环
    split_true = []
    split_pred = []

    print("🚀 Starting Evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # batch: 0:sp, 1:tm, 2:target, 3:mit_mask, 4:aux(ignore), 5:gold_mask
            b_sp = batch[0].to(device)
            b_tm = batch[1].to(device)
            b_target = batch[2].to(device)
            b_mit_mask = batch[3].to(device)
            b_gold_mask = batch[5].to(device)

            # 预测
            final, _, _ = model(b_sp, b_tm)

            # 反归一化函数
            def inverse(t):
                if t is None: return None
                # t: (Batch, 1, H, W) -> numpy
                arr = t.cpu().numpy().reshape(t.shape[0], -1)
                # inverse_transform 对每一行进行反归一化
                inv = scaler_y.inverse_transform(arr)
                return inv.reshape(t.shape[0], H, W)

            # 批量反归一化
            map_pred_batch = inverse(final)  # (Batch, H, W)
            map_true_batch = b_target.cpu().numpy().reshape(-1, H, W)

            mask_mit_batch = b_mit_mask.cpu().numpy().reshape(-1, H, W).astype(bool)
            mask_gold_batch = b_gold_mask.cpu().numpy().reshape(-1, H, W).astype(bool)

            # 交集掩码 (MIT & GOLD)
            mask_inter_batch = mask_mit_batch & mask_gold_batch

            # 【关键】应用区域掩码
            # static_region_mask 是 (H, W)，自动广播到 (Batch, H, W)
            final_mask_batch = mask_inter_batch & static_region_mask

            # 收集有效点
            for j in range(len(map_pred_batch)):
                mask = final_mask_batch[j]
                if np.any(mask):
                    split_true.append(map_true_batch[j][mask])
                    split_pred.append(map_pred_batch[j][mask])

    # 9. 计算并输出最终结果
    if split_true:
        all_true = np.concatenate(split_true)
        all_pred = np.concatenate(split_pred)
        mse = mean_squared_error(all_true, all_pred)
        rmse = np.sqrt(mse)

        # ！！！这是脚本的最终输出！！！
        print(f"FINAL_RESULT_RMSE:{rmse:.4f}")
    else:
        print("FINAL_RESULT_RMSE:NaN")


if __name__ == "__main__":
    main()