import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.utils.common import setup_gpu
from src.utils.metrics import masked_loss, GradientLoss, SmoothnessLoss
from src.models.fusion import TECFusionCNNModel
from src.utils.options import parse_args
from src.utils.logger import Logger


def save_checkpoint(state, save_dir, filename):
    path = os.path.join(save_dir, filename)
    torch.save(state, path)


def load_and_process_data(dataset_path):
    """辅助函数：加载pth文件并转换为Numpy数组 (解包逻辑)"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到数据集: {dataset_path}")

    # 兼容加载方式
    try:
        samples = torch.load(dataset_path, weights_only=False)
    except TypeError:
        samples = torch.load(dataset_path)

    X_spatial_list, X_time_list, y_list = [], [], []
    mit_mask_list, gold_rad_mask_list = [], []

    for sample in samples:
        X_spatial_list.append(sample[0].unsqueeze(0))
        X_time_list.append(sample[1].unsqueeze(0))
        y_list.append(sample[2].unsqueeze(0).unsqueeze(1))
        mit_mask_list.append(sample[3].unsqueeze(0).unsqueeze(1))
        gold_rad_mask_list.append(sample[5].unsqueeze(0).unsqueeze(1))

    X_spatial = torch.cat(X_spatial_list, dim=0).numpy()
    X_time = torch.cat(X_time_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()
    mit_masks = torch.cat(mit_mask_list, dim=0).numpy()
    gold_rad_masks = torch.cat(gold_rad_mask_list, dim=0).numpy()

    if y.ndim == 3: y = y[:, np.newaxis, ...]

    return X_spatial, X_time, y, mit_masks, gold_rad_masks


def run_server_training():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = setup_gpu()

    # 1. 复制默认配置
    params = Config.MODEL_PARAMS.copy()

    # 2. 用命令行参数覆盖 Config (通用参数)
    if args.epochs: params['epochs'] = args.epochs
    if args.batch_size: params['batch_size'] = args.batch_size
    if args.lr: params['learning_rate'] = args.lr
    params['random_state'] = args.seed

    if args.model_mode: params['model_mode'] = args.model_mode
    if args.use_spatial_attention is not None:
        params['use_spatial_attention'] = bool(args.use_spatial_attention)

    # 3. 【关键修改】动态配置陆地遮挡参数
    # 默认从 Config 读取，如果命令行有传参则覆盖
    enable_masking = getattr(Config, 'ENABLE_LAND_MASKING', False)
    if args.land_masking is not None:
        enable_masking = bool(args.land_masking)

    # 将其写入 params 以便保存到模型文件
    params['enable_land_masking'] = enable_masking

    # 解析区域字符串 "-70,-60,-30,-10"
    mask_region = getattr(Config, 'LAND_MASK_REGION', None)
    if args.mask_region:
        try:
            vals = list(map(float, args.mask_region.split(',')))
            mask_region = {
                'lon_min': vals[0], 'lon_max': vals[1],
                'lat_min': vals[2], 'lat_max': vals[3]
            }
            # 同时也必须强制开启 masking，防止手误
            enable_masking = True
            params['enable_land_masking'] = True
        except Exception as e:
            print(f"❌ 解析 mask_region 失败: {e}")
            sys.exit(1)

    params['land_mask_region'] = mask_region

    # === 必须反向更新 Config，因为后续代码逻辑（如遮挡）可能直接读 Config ===
    Config.ENABLE_LAND_MASKING = enable_masking
    Config.LAND_MASK_REGION = mask_region

    # === 继续原有流程 ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = setup_gpu()

    if args.epochs: params['epochs'] = args.epochs
    if args.batch_size: params['batch_size'] = args.batch_size
    if args.lr: params['learning_rate'] = args.lr
    params['random_state'] = args.seed

    log_root = os.path.join(project_root, 'logs')
    logger = Logger(log_root, args.exp_name)
    logger.log(f"=== 服务器训练启动: {args.exp_name} ===")

    current_mode = params.get('model_mode', 'fusion')
    logger.log(f"当前模型模式: {current_mode}")
    logger.log(f"陆地遮挡: {getattr(Config, 'ENABLE_LAND_MASKING', False)}")

    # ================= 数据加载部分 =================
    base_path = args.dataset_path if args.dataset_path else Config.DATASET_PATH
    scaler_X_spatial = StandardScaler()
    scaler_X_time = StandardScaler()
    scaler_y = StandardScaler()

    if Config.USE_PERIODIC_SPLIT:
        logger.log("🔀 [Mode] 使用周期性划分数据集 (Train/Val/Test 分离)")
        base_name, ext = os.path.splitext(base_path)
        train_path = f"{base_name}_train{ext}"
        val_path = f"{base_name}_val{ext}"

        X_sp_train_raw, X_tm_train_raw, y_train_raw, mask_mit_train, mask_gold_train = load_and_process_data(train_path)
        X_sp_val_raw, X_tm_val_raw, y_val_raw, mask_mit_val, mask_gold_val = load_and_process_data(val_path)
        logger.log(f"Loaded Train: {len(X_sp_train_raw)}, Val: {len(X_sp_val_raw)}")

        logger.log("Fitting Scalers on Train Set...")
        flat_train_sp = X_sp_train_raw.reshape(len(X_sp_train_raw), -1)
        X_sp_train = scaler_X_spatial.fit_transform(flat_train_sp).reshape(X_sp_train_raw.shape)
        flat_val_sp = X_sp_val_raw.reshape(len(X_sp_val_raw), -1)
        X_sp_val = scaler_X_spatial.transform(flat_val_sp).reshape(X_sp_val_raw.shape)

        X_tm_train = scaler_X_time.fit_transform(X_tm_train_raw)
        X_tm_val = scaler_X_time.transform(X_tm_val_raw)

        flat_train_y = y_train_raw.reshape(len(y_train_raw), -1)
        y_train = scaler_y.fit_transform(flat_train_y).reshape(y_train_raw.shape)
        flat_val_y = y_val_raw.reshape(len(y_val_raw), -1)
        y_val = scaler_y.transform(flat_val_y).reshape(y_val_raw.shape)

        train_subset = TensorDataset(
            torch.FloatTensor(X_sp_train), torch.FloatTensor(X_tm_train),
            torch.FloatTensor(y_train), torch.BoolTensor(mask_mit_train),
            torch.BoolTensor(mask_gold_train)
        )
        val_subset = TensorDataset(
            torch.FloatTensor(X_sp_val), torch.FloatTensor(X_tm_val),
            torch.FloatTensor(y_val), torch.BoolTensor(mask_mit_val),
            torch.BoolTensor(mask_gold_val)
        )
    else:
        logger.log("⏹ [Mode] 使用随机划分数据集")
        X_spatial, X_time, y, mit_masks, gold_rad_masks = load_and_process_data(base_path)
        pass

    train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False, pin_memory=True)

    # ================= 模型初始化 =================
    sample_x, _, _, _, _ = train_subset[0]
    model_config = params.copy()
    model_config['input_shape'] = sample_x.shape[1:]
    model_config['input_channels'] = sample_x.shape[0]
    model_config['time_features_dim'] = X_tm_train_raw.shape[1] if Config.USE_PERIODIC_SPLIT else X_tm_train.shape[1]

    model = TECFusionCNNModel(model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # 初始化损失函数
    grad_criterion = GradientLoss(device)
    smooth_criterion = SmoothnessLoss().to(device)
    grad_weight = 0.1

    # 断点续训逻辑
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    resume_path = os.path.join(logger.get_save_dir(), 'last_checkpoint.pth')
    if args.resume and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.log(f"恢复成功，从 Epoch {start_epoch} 继续")

    # ================= 训练循环 =================
    for epoch in range(start_epoch, params['epochs']):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            b_spatial, b_time, b_y, b_mit_mask, b_gold_mask = [b.to(device) for b in batch]
            optimizer.zero_grad()

            outputs, _, _ = model(b_spatial, b_time)
            intersection_mask = b_mit_mask & b_gold_mask

            # 【陆地遮挡实验逻辑】
            if getattr(Config, 'ENABLE_LAND_MASKING', False):
                region = Config.LAND_MASK_REGION
                H, W = intersection_mask.shape[2], intersection_mask.shape[3]
                lon_step = (Config.MASK_LON_MAX - Config.MASK_LON_MIN) / W
                lat_step = (Config.MASK_LAT_MAX - Config.MASK_LAT_MIN) / H

                c_min = int((region['lon_min'] - Config.MASK_LON_MIN) / lon_step)
                c_max = int((region['lon_max'] - Config.MASK_LON_MIN) / lon_step)
                r_min = int((region['lat_min'] - Config.MASK_LAT_MIN) / lat_step)
                r_max = int((region['lat_max'] - Config.MASK_LAT_MIN) / lat_step)

                c_min, c_max = max(0, min(W, c_min)), max(0, min(W, c_max))
                r_min, r_max = max(0, min(H, r_min)), max(0, min(H, r_max))

                if c_max > c_min and r_max > r_min:
                    intersection_mask[:, :, r_min:r_max, c_min:c_max] = False

            loss_pixel = masked_loss(outputs, b_y, intersection_mask, beta=params['loss_beta'])
            loss_grad = grad_criterion(outputs, b_y, intersection_mask)

            loss_smooth = torch.tensor(0.0, device=device)
            if params.get('enable_smooth_loss', False):
                loss_smooth = smooth_criterion(outputs)
                loss = loss_pixel + (grad_weight * loss_grad) + (params.get('smooth_loss_weight', 0.05) * loss_smooth)
            else:
                loss = loss_pixel + (grad_weight * loss_grad)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_spatial.size(0)

        train_loss /= len(train_subset)

        # ================= 验证循环 =================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b_spatial, b_time, b_y, b_mit_mask, b_gold_mask = [b.to(device) for b in batch]
                outputs, _, _ = model(b_spatial, b_time)
                intersection_mask = b_mit_mask & b_gold_mask

                v_loss_pixel = masked_loss(outputs, b_y, intersection_mask, beta=params['loss_beta'])
                v_loss_grad = grad_criterion(outputs, b_y, intersection_mask)

                if params.get('enable_smooth_loss', False):
                    v_loss_smooth = smooth_criterion(outputs)
                    batch_loss = v_loss_pixel + (grad_weight * v_loss_grad) + (
                                params.get('smooth_loss_weight', 0.05) * v_loss_smooth)
                else:
                    batch_loss = v_loss_pixel + (grad_weight * v_loss_grad)

                val_loss += batch_loss.item() * b_spatial.size(0)

        val_loss /= len(val_subset)
        logger.log_metric('Loss/Train', train_loss, epoch)
        logger.log_metric('Loss/Val', val_loss, epoch)

        if (epoch + 1) % args.log_interval == 0:
            logger.log(f"Epoch [{epoch + 1}/{params['epochs']}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'scaler_X_spatial': scaler_X_spatial,
            'scaler_X_time': scaler_X_time,
            'scaler_y': scaler_y
        }, logger.get_save_dir(), 'last_checkpoint.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': model_config,  # 这里会保存包含新参数的 config
                'scaler_X_spatial': scaler_X_spatial,
                'scaler_X_time': scaler_X_time,
                'scaler_y': scaler_y,
                'metrics': {'val_loss': val_loss}
            }, logger.get_save_dir(), 'best_model.pth')
            logger.log(f"--> 保存新的最佳模型 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= params['early_stopping_patience']:
                logger.log(f"🛑 触发早停 (Early Stopping) at Epoch {epoch + 1}")
                break

    logger.log("=== 训练结束 ===")
    logger.close()


if __name__ == "__main__":
    run_server_training()