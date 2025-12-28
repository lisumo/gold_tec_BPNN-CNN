import sys
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.utils.common import setup_gpu
from src.utils.metrics import masked_loss, calculate_residual_stats
from src.utils.visualization import (
    plot_training_validation_loss, plot_prediction_vs_truth, plot_residual_histogram,
    plot_true_tec_distribution, plot_cnn_prediction, plot_bpnn_background,
    plot_fusion_result, plot_cnn_residual, plot_fusion_residual
)
from src.models.fusion import TECFusionCNNModel


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


def train_pipeline():
    # 1. 设置设备
    device = setup_gpu()
    params = Config.MODEL_PARAMS

    print(f"当前模式: {'周期性划分 (Periodic Split)' if Config.USE_PERIODIC_SPLIT else '随机划分 (Random Split)'}")

    # 2. 数据加载与预处理
    scaler_X_spatial = StandardScaler()
    scaler_X_time = StandardScaler()
    scaler_y = StandardScaler()

    if Config.USE_PERIODIC_SPLIT:
        # === 周期性划分模式 (加载拆分好的文件) ===
        base_name, ext = os.path.splitext(Config.DATASET_PATH)
        train_path = f"{base_name}_train{ext}"
        val_path = f"{base_name}_val{ext}"
        test_path = f"{base_name}_test{ext}"

        print(f"正在加载训练集: {train_path}")
        X_sp_train, X_tm_train, y_train, mask_mit_train, mask_gold_train = load_and_process_data(train_path)
        print(f"正在加载验证集: {val_path}")
        X_sp_val, X_tm_val, y_val, mask_mit_val, mask_gold_val = load_and_process_data(val_path)
        print(f"正在加载测试集: {test_path}")
        X_sp_test, X_tm_test, y_test, mask_mit_test, mask_gold_test = load_and_process_data(test_path)

        # 拟合标准化器 (仅用训练集)
        print("执行标准化 (Fit on Train)...")
        flat_train_sp = X_sp_train.reshape(len(X_sp_train), -1)
        X_spatial_train = scaler_X_spatial.fit_transform(flat_train_sp).reshape(X_sp_train.shape)

        flat_val_sp = X_sp_val.reshape(len(X_sp_val), -1)
        X_spatial_val = scaler_X_spatial.transform(flat_val_sp).reshape(X_sp_val.shape)

        flat_test_sp = X_sp_test.reshape(len(X_sp_test), -1)
        X_spatial_test = scaler_X_spatial.transform(flat_test_sp).reshape(X_sp_test.shape)

        X_time_train = scaler_X_time.fit_transform(X_tm_train)
        X_time_val = scaler_X_time.transform(X_tm_val)
        X_time_test = scaler_X_time.transform(X_tm_test)

        flat_train_y = y_train.reshape(len(y_train), -1)
        y_train_scaled = scaler_y.fit_transform(flat_train_y).reshape(y_train.shape)

        flat_val_y = y_val.reshape(len(y_val), -1)
        y_val_scaled = scaler_y.transform(flat_val_y).reshape(y_val.shape)

        flat_test_y = y_test.reshape(len(y_test), -1)
        y_test_scaled = scaler_y.transform(flat_test_y).reshape(y_test.shape)

        # 构建 Dataset
        train_ds = TensorDataset(
            torch.FloatTensor(X_spatial_train).to(device), torch.FloatTensor(X_time_train).to(device),
            torch.FloatTensor(y_train_scaled).to(device), torch.BoolTensor(mask_mit_train).to(device),
            torch.BoolTensor(mask_gold_train).to(device)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_spatial_val).to(device), torch.FloatTensor(X_time_val).to(device),
            torch.FloatTensor(y_val_scaled).to(device), torch.BoolTensor(mask_mit_val).to(device),
            torch.BoolTensor(mask_gold_val).to(device)
        )

        # 为绘图保留原始测试集数据
        y_test_orig = y_test
        mit_masks_test = mask_mit_test
        gold_rad_masks_test = mask_gold_test

        # 测试集 Tensor
        X_spatial_test_tensor = torch.FloatTensor(X_spatial_test).to(device)
        X_time_test_tensor = torch.FloatTensor(X_time_test).to(device)

    else:
        # === 随机划分模式 (旧逻辑) ===
        print(f"加载完整数据集: {Config.DATASET_PATH}")
        X_spatial, X_time, y, mit_masks, gold_rad_masks = load_and_process_data(Config.DATASET_PATH)

        # 划分索引
        indices = np.arange(len(X_spatial))
        np.random.seed(params['random_state'])
        np.random.shuffle(indices)

        test_size = int(params['test_size'] * len(indices))
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        # 切分
        X_spatial_train = X_spatial[train_indices]
        X_spatial_test = X_spatial[test_indices]
        X_time_train = X_time[train_indices]
        X_time_test = X_time[test_indices]
        y_train = y[train_indices]
        y_test_orig = y[test_indices]  # 保留原始值用于评估

        mit_masks_train = mit_masks[train_indices]
        mit_masks_test = mit_masks[test_indices]
        gold_rad_masks_train = gold_rad_masks[train_indices]
        gold_rad_masks_test = gold_rad_masks[test_indices]

        # 标准化
        print("执行标准化...")
        flat_train_sp = X_spatial_train.reshape(len(train_indices), -1)
        X_spatial_train = scaler_X_spatial.fit_transform(flat_train_sp).reshape(X_spatial_train.shape)

        flat_test_sp = X_spatial_test.reshape(len(test_indices), -1)
        X_spatial_test = scaler_X_spatial.transform(flat_test_sp).reshape(X_spatial_test.shape)

        X_time_train = scaler_X_time.fit_transform(X_time_train)
        X_time_test = scaler_X_time.transform(X_time_test)

        flat_train_y = y_train.reshape(len(train_indices), -1)
        y_train_scaled = scaler_y.fit_transform(flat_train_y).reshape(y_train.shape)

        flat_test_y = y_test_orig.reshape(len(test_indices), -1)
        y_test_scaled = scaler_y.transform(flat_test_y).reshape(y_test_orig.shape)

        # 构建 Dataset (先构建全量训练集，再分验证集)
        full_train_ds = TensorDataset(
            torch.FloatTensor(X_spatial_train).to(device),
            torch.FloatTensor(X_time_train).to(device),
            torch.FloatTensor(y_train_scaled).to(device),
            torch.BoolTensor(mit_masks_train).to(device),
            torch.BoolTensor(gold_rad_masks_train).to(device)
        )

        val_len = int(params['val_size'] * len(full_train_ds))
        train_len = len(full_train_ds) - val_len
        train_ds, val_ds = random_split(
            full_train_ds, [train_len, val_len],
            generator=torch.Generator().manual_seed(params['random_state'])
        )

        X_spatial_test_tensor = torch.FloatTensor(X_spatial_test).to(device)
        X_time_test_tensor = torch.FloatTensor(X_time_test).to(device)

    # 3. DataLoader
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)

    # 4. 初始化模型
    # 获取输入维度
    if Config.USE_PERIODIC_SPLIT:
        input_shape = X_spatial_train.shape[2:]
        input_channels = X_spatial_train.shape[1]
        time_dim = X_time_train.shape[1]
    else:
        sample_x = X_spatial_train[0]
        input_shape = sample_x.shape[1:]
        input_channels = sample_x.shape[0]
        time_dim = X_time_train.shape[1]

    model_config = params.copy()
    model_config['input_shape'] = input_shape
    model_config['input_channels'] = input_channels
    model_config['time_features_dim'] = time_dim

    model = TECFusionCNNModel(model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # 5. 训练循环
    print("\n开始训练...")
    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            b_spatial, b_time, b_y, b_mit_mask, b_gold_mask = batch
            optimizer.zero_grad()

            # 统一调用
            outputs, _, _ = model(b_spatial, b_time)

            intersection_mask = b_mit_mask & b_gold_mask
            loss = masked_loss(outputs, b_y, intersection_mask, beta=params['loss_beta'])

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_spatial.size(0)

        train_loss /= len(train_ds)
        history['loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                b_spatial, b_time, b_y, b_mit_mask, b_gold_mask = batch

                outputs, _, _ = model(b_spatial, b_time)

                intersection_mask = b_mit_mask & b_gold_mask
                loss = masked_loss(outputs, b_y, intersection_mask, beta=params['loss_beta'])
                val_loss += loss.item() * b_spatial.size(0)

        val_loss /= len(val_ds)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % params['print_interval'] == 0:
            print(f"Epoch [{epoch + 1}/{params['epochs']}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= params['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 6. 保存与评估
    if best_weights:
        model.load_state_dict(best_weights)

    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X_spatial': scaler_X_spatial,
        'scaler_X_time': scaler_X_time,
        'scaler_y': scaler_y,
        'config': model_config
    }, Config.MODEL_SAVE_PATH)
    print(f"模型已保存: {Config.MODEL_SAVE_PATH}")

    # 7. 最终评估 (使用测试集)
    model.eval()
    with torch.no_grad():
        pred_tensor, cnn_out_tensor, bpnn_out_tensor = model(X_spatial_test_tensor, X_time_test_tensor)
        y_pred = pred_tensor.cpu().numpy()

    # 反归一化预测值
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(y_pred.shape[0], -1)).reshape(y_pred.shape)

    # y_test_orig 已经是反归一化的 (对于周期性划分是直接加载的原始值，对于随机划分是保留的副本)
    if Config.USE_PERIODIC_SPLIT:
        # 在周期性模式下，y_test 加载时被标准化了，所以这里也要反归一化回去，或者直接用原始加载的y
        # 为了保险，我们用 scaler 反变换 y_test_scaled
        y_test_orig = scaler_y.inverse_transform(y_test_scaled.reshape(y_test_scaled.shape[0], -1)).reshape(
            y_test_scaled.shape)

    # 统计指标
    y_test_valid_points = []
    y_pred_valid_points = []

    for i in range(len(y_test_orig)):
        mask = mit_masks_test[i, 0] & gold_rad_masks_test[i, 0]
        y_test_valid_points.extend(y_test_orig[i, 0][mask].flatten())
        y_pred_valid_points.extend(y_pred_orig[i, 0][mask].flatten())

    y_test_valid_points = np.array(y_test_valid_points)
    y_pred_valid_points = np.array(y_pred_valid_points)

    if len(y_test_valid_points) > 0:
        mse = mean_squared_error(y_test_valid_points, y_pred_valid_points)
        print(f"\n测试集评估 (有效区域):")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        print(f"MAE: {mean_absolute_error(y_test_valid_points, y_pred_valid_points):.4f}")
        print(f"R2: {r2_score(y_test_valid_points, y_pred_valid_points):.4f}")

        # 绘图
        plot_training_validation_loss(history)
        residuals, stats = calculate_residual_stats(y_test_valid_points, y_pred_valid_points)
        plot_residual_histogram(residuals)
        plot_prediction_vs_truth(y_test_valid_points, y_pred_valid_points)

        # 随机样本可视化
        sample_idx = np.random.randint(0, len(y_test_orig))

        # 反归一化子模型输出
        if cnn_out_tensor is not None:
            cnn_orig = scaler_y.inverse_transform(cnn_out_tensor.cpu().numpy().reshape(len(y_test_orig), -1)).reshape(
                y_test_orig.shape)
        else:
            cnn_orig = np.zeros_like(y_pred_orig)

        if bpnn_out_tensor is not None:
            bpnn_orig = scaler_y.inverse_transform(bpnn_out_tensor.cpu().numpy().reshape(len(y_test_orig), -1)).reshape(
                y_test_orig.shape)
        else:
            bpnn_orig = np.zeros_like(y_pred_orig)

        # 统一色标
        true_tec = y_test_orig[sample_idx, 0]
        true_mask = mit_masks_test[sample_idx, 0]
        masked_true = np.where(true_mask, true_tec, np.nan)
        # 防止全nan报错
        if np.isnan(masked_true).all():
            vmin, vmax = 0, 100
        else:
            vmin, vmax = np.nanmin(masked_true), np.nanmax(masked_true)

        plot_true_tec_distribution(y_test_orig, mit_masks_test, sample_idx)

        if params.get('model_mode') in ['fusion', 'cnn']:
            plot_cnn_prediction(cnn_orig, gold_rad_masks_test, sample_idx, vmin, vmax)
            plot_cnn_residual(cnn_orig, y_test_orig, mit_masks_test, gold_rad_masks_test, sample_idx)

        if params.get('model_mode') in ['fusion', 'bpnn']:
            plot_bpnn_background(bpnn_orig, gold_rad_masks_test, sample_idx, vmin, vmax)

        plot_fusion_result(y_pred_orig, gold_rad_masks_test, sample_idx, vmin, vmax)
        plot_fusion_residual(y_pred_orig, y_test_orig, mit_masks_test, gold_rad_masks_test, sample_idx)


if __name__ == "__main__":
    train_pipeline()