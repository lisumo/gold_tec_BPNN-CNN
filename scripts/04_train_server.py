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
from src.utils.metrics import masked_loss, GradientLoss
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

    params = Config.MODEL_PARAMS.copy()
    if args.epochs: params['epochs'] = args.epochs
    if args.batch_size: params['batch_size'] = args.batch_size
    if args.lr: params['learning_rate'] = args.lr
    params['random_state'] = args.seed

    log_root = os.path.join(project_root, 'logs')
    logger = Logger(log_root, args.exp_name)
    logger.log(f"=== 服务器训练启动: {args.exp_name} ===")

    # 打印当前模式
    current_mode = params.get('model_mode', 'fusion')
    logger.log(f"当前模型模式: {current_mode}")

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
        # 随机划分逻辑 (兼容旧代码)
        logger.log("⏹ [Mode] 使用随机划分数据集")
        X_spatial, X_time, y, mit_masks, gold_rad_masks = load_and_process_data(base_path)

        # ... (此处省略随机划分的具体实现，建议尽量使用周期性划分) ...
        # 如果需要完整兼容，请保留您原来的随机划分代码块
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

    # 初始化梯度损失
    grad_criterion = GradientLoss(device)
    grad_weight = 0.1  # 梯度损失权重

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

            # 【核心修改】统一调用接口，无论是什么模式，model都会返回三个值
            # outputs 是最终用于计算 loss 的预测值 (Fusion / CNN / BPNN)
            outputs, _, _ = model(b_spatial, b_time)

            # 计算交集掩码
            intersection_mask = b_mit_mask & b_gold_mask

            # 1. 基础像素损失 (SmoothL1)
            loss_pixel = masked_loss(outputs, b_y, intersection_mask, beta=params['loss_beta'])

            # 2. 结构梯度损失
            loss_grad = grad_criterion(outputs, b_y, intersection_mask)

            # 总损失
            loss = loss_pixel + grad_weight * loss_grad

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

                # 【核心修改】统一调用接口
                outputs, _, _ = model(b_spatial, b_time)

                intersection_mask = b_mit_mask & b_gold_mask

                v_loss_pixel = masked_loss(outputs, b_y, intersection_mask, beta=params['loss_beta'])
                v_loss_grad = grad_criterion(outputs, b_y, intersection_mask)

                batch_loss = v_loss_pixel + grad_weight * v_loss_grad
                val_loss += batch_loss.item() * b_spatial.size(0)

        val_loss /= len(val_subset)

        logger.log_metric('Loss/Train', train_loss, epoch)
        logger.log_metric('Loss/Val', val_loss, epoch)

        if (epoch + 1) % args.log_interval == 0:
            logger.log(f"Epoch [{epoch + 1}/{params['epochs']}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 保存 Last Checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'scaler_X_spatial': scaler_X_spatial,
            'scaler_X_time': scaler_X_time,
            'scaler_y': scaler_y
        }, logger.get_save_dir(), 'last_checkpoint.pth')

        # 保存 Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': model_config,
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