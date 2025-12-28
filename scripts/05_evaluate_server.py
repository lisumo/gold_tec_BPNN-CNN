import matplotlib

# 1. 强制设置 Agg 后端 (必须在 pyplot 导入前)
matplotlib.use('Agg')

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Cartopy 导入 (可选) ===
try:
    import cartopy.crs as ccrs
    from cartopy.feature import NaturalEarthFeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.utils.common import setup_gpu
from src.models.fusion import TECFusionCNNModel
from src.utils.options import parse_args

# 2. 强制设置英文字体 (解决 SimHei 报错)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# === 自定义 Dataset ===
class EvaluatorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # 解包 (支持变长)
        # 0:sp, 1:tm, 2:tgt, 3:mit, 4:aux, 5:gold
        spatial = item[0]
        time_feat = item[1]
        target = item[2]
        mit_mask = item[3]
        gold_mask = item[5]
        # 尝试获取时间字符串
        time_str = item[6] if len(item) > 6 else ""
        return spatial, time_feat, target, mit_mask, gold_mask, time_str


def add_map_features(ax):
    if not HAS_CARTOPY: return
    coastline = NaturalEarthFeature('physical', 'coastline', '50m', edgecolor='black', facecolor='none')
    ax.add_feature(coastline, linewidth=0.8)
    borders = NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor='none')
    ax.add_feature(borders, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


def evaluate_pipeline():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = setup_gpu()

    log_dir = os.path.join(project_root, 'logs', args.exp_name)
    # 自动寻找最新的实验目录 (如果指定的exp_name里没有模型)
    model_path = os.path.join(log_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        logs_root = os.path.join(project_root, 'logs')
        if os.path.exists(logs_root):
            exps = sorted([os.path.join(logs_root, d) for d in os.listdir(logs_root) if
                           os.path.isdir(os.path.join(logs_root, d))], key=os.path.getmtime)
            if exps:
                model_path = os.path.join(exps[-1], 'best_model.pth')
                log_dir = exps[-1]
                print(f"Redirecting to latest experiment: {model_path}")

    eval_dir = os.path.join(log_dir, 'evaluation_results')
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    # 3. 兼容加载 (weights_only=False)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint['config']
    scaler_y = checkpoint['scaler_y']
    scaler_X_spatial = checkpoint.get('scaler_X_spatial') or checkpoint['scaler_X_spatial']
    scaler_X_time = checkpoint.get('scaler_X_time') or checkpoint['scaler_X_time']

    # 打印当前模式
    print(f"Model Mode: {model_config.get('model_mode', 'fusion')}")

    # 加载数据
    base_path = args.dataset_path if args.dataset_path else Config.DATASET_PATH
    if Config.USE_PERIODIC_SPLIT:
        base, ext = os.path.splitext(base_path)
        test_path = f"{base}_test{ext}"
    else:
        test_path = base_path

    print(f"Loading test data: {test_path}")
    try:
        raw_samples = torch.load(test_path, weights_only=False)
    except TypeError:
        raw_samples = torch.load(test_path)

    # 预处理数据
    processed_samples = []
    print("Preprocessing data...")
    for s in raw_samples:
        # Spatial Norm
        sp = s[0].numpy()
        C, H, W = sp.shape
        sp_flat = sp.reshape(1, -1)
        sp_norm = scaler_X_spatial.transform(sp_flat).reshape(C, H, W)

        # Time Norm
        tm = s[1].numpy().reshape(1, -1)
        tm_norm = scaler_X_time.transform(tm).reshape(-1)

        new_sample = [
            torch.FloatTensor(sp_norm),
            torch.FloatTensor(tm_norm),
            s[2], s[3], s[4], s[5]
        ]
        if len(s) > 6: new_sample.append(s[6])
        processed_samples.append(tuple(new_sample))

    dataset = EvaluatorDataset(processed_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = TECFusionCNNModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Calculating valid pixels for selection...")
    valid_counts = []
    for i in range(len(raw_samples)):
        # Intersection mask count
        mask = raw_samples[i][3] & raw_samples[i][5]
        valid_counts.append(mask.sum().item())

    # 选前 20 个样本绘图
    top_indices = np.argsort(valid_counts)[-20:][::-1]
    top_indices_set = set(top_indices)

    # 生成网格
    lon_arr = np.linspace(Config.MASK_LON_MIN, Config.MASK_LON_MAX, model_config['input_shape'][1])
    lat_arr = np.linspace(Config.MASK_LAT_MIN, Config.MASK_LAT_MAX, model_config['input_shape'][0])
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    all_true_values = []
    all_pred_values = []

    print("Running inference (Statistics & Plotting)...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            b_sp, b_tm, b_target, b_mit_mask, b_gold_mask = [x.to(device) for x in batch[:5]]
            time_str = batch[5][0] if len(batch) > 5 else f"Sample {i}"

            # 4. 统一调用 model (返回三元组)
            final, cnn, bpnn = model(b_sp, b_tm)

            # 5. 定义反归一化函数 (含 None 检查)
            def inverse(t):
                if t is None: return None
                arr = t.cpu().numpy().reshape(1, -1)
                return scaler_y.inverse_transform(arr).reshape(model_config['input_shape'])

            map_pred = inverse(final)
            map_true = b_target.cpu().numpy().reshape(model_config['input_shape'])

            mask_mit = b_mit_mask.cpu().numpy().reshape(model_config['input_shape']).astype(bool)
            mask_gold = b_gold_mask.cpu().numpy().reshape(model_config['input_shape']).astype(bool)
            mask_inter = mask_mit & mask_gold

            # 收集统计
            valid_true = map_true[mask_inter]
            valid_pred = map_pred[mask_inter]
            if len(valid_true) > 0:
                all_true_values.append(valid_true)
                all_pred_values.append(valid_pred)

            # 绘图逻辑 (仅针对 Top 20)
            if i in top_indices_set:
                map_cnn = inverse(cnn)
                map_bpnn = inverse(bpnn)

                # 6. 安全创建可视化矩阵 (None -> NaN)
                empty_map = np.full(model_config['input_shape'], np.nan)

                vis_true = np.where(mask_mit, map_true, np.nan)
                vis_pred = np.where(mask_gold, map_pred, np.nan)

                # 针对可能缺失的分支使用 empty_map 填充
                vis_cnn = np.where(mask_gold, map_cnn, np.nan) if map_cnn is not None else empty_map
                vis_bpnn = np.where(mask_gold, map_bpnn, np.nan) if map_bpnn is not None else empty_map

                vis_diff = np.where(mask_inter, map_pred - map_true, np.nan)

                # 7. 确保 vmin 计算在 vis_true 定义之后 (在同一个缩进块内)
                vmin = np.nanmin(vis_true)
                vmax = np.nanmax(vis_true)
                diff_max = np.nanmax(np.abs(vis_diff)) if np.any(~np.isnan(vis_diff)) else 1.0

                if HAS_CARTOPY:
                    proj = ccrs.PlateCarree()
                    fig, axes = plt.subplots(1, 5, figsize=(25, 5), subplot_kw={'projection': proj})
                else:
                    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

                def plot_ax(ax, data, title, cmap, vmin, vmax):
                    # 如果全为 NaN，imshow 可能会报警告，这里不处理也没事
                    if HAS_CARTOPY:
                        im = ax.pcolormesh(lon_grid, lat_grid, data, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
                                           shading='auto')
                        add_map_features(ax)
                    else:
                        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                    ax.set_title(title, fontsize=10)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                plot_ax(axes[0], vis_true, f"Ground Truth (MIT)\n{time_str}", 'viridis', vmin, vmax)
                plot_ax(axes[1], vis_bpnn, f"BPNN Component\n(Background)", 'viridis', vmin, vmax)
                plot_ax(axes[2], vis_cnn, f"CNN Branch Output\n(Spatial Feature)", 'viridis', vmin, vmax)
                plot_ax(axes[3], vis_pred, f"Final Prediction\n(Fused)", 'viridis', vmin, vmax)
                plot_ax(axes[4], vis_diff, f"Difference\n(RMSE={np.sqrt(np.nanmean(vis_diff ** 2)):.2f})", 'RdBu_r',
                        -diff_max, diff_max)

                plt.tight_layout()
                safe_name = time_str.replace(':', '-').replace(' ', '_') if time_str else f"sample_{i}"
                plt.savefig(os.path.join(eval_dir, f"Analysis_{safe_name}.png"), dpi=150)
                plt.close()

    # === 全局指标 ===
    print("Calculating Global Metrics...")
    if len(all_true_values) > 0:
        y_true_all = np.concatenate(all_true_values)
        y_pred_all = np.concatenate(all_pred_values)

        mse = mean_squared_error(y_true_all, y_pred_all)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)
        nmae = mae / (np.mean(np.abs(y_true_all)) + 1e-6)

        metrics_str = (
            f"=== Global Evaluation Results (Test Set) ===\n"
            f"Mode: {model_config.get('model_mode', 'fusion')}\n"
            f"Total Samples: {len(raw_samples)}\n"
            f"Total Valid Pixels: {len(y_true_all)}\n"
            f"RMSE: {rmse:.4f} TECU\n"
            f"MAE : {mae:.4f} TECU\n"
            f"R2  : {r2:.4f}\n"
            f"NMAE: {nmae:.2%}\n"
        )
        print(metrics_str)

        with open(os.path.join(eval_dir, 'metrics.txt'), 'w') as f:
            f.write(metrics_str)

        # 散点图
        plt.figure(figsize=(8, 8))
        if len(y_true_all) > 50000:
            indices = np.random.choice(len(y_true_all), 50000, replace=False)
            plt.scatter(y_true_all[indices], y_pred_all[indices], alpha=0.1, s=1, c='blue')
        else:
            plt.scatter(y_true_all, y_pred_all, alpha=0.1, s=1, c='blue')

        plt.plot([y_true_all.min(), y_true_all.max()], [y_true_all.min(), y_true_all.max()], 'r--', lw=2)
        plt.xlabel('True TEC (TECU)')
        plt.ylabel('Predicted TEC (TECU)')
        plt.title(f'Global Prediction vs Truth (RMSE={rmse:.2f})')
        plt.grid(True)
        plt.savefig(os.path.join(eval_dir, 'global_scatter_plot.png'), dpi=150)
        plt.close()

    else:
        print("Warning: No valid intersection pixels found in test set!")

    print(f"✅ Evaluation complete. Results saved to {eval_dir}")


if __name__ == "__main__":
    evaluate_pipeline()