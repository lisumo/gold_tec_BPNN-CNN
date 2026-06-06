# src/utils/metrics.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F

def calculate_residual_stats(y_true, y_pred):
    """计算残差统计信息"""
    # 将张量展平为一维数组
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    residuals = y_pred_flat - y_true_flat

    stats = {
        '残差均值': np.mean(residuals),
        '残差中位数': np.median(residuals),
        '残差标准差': np.std(residuals),
        '残差最小值': np.min(residuals),
        '残差最大值': np.max(residuals),
        '残差绝对值均值': np.mean(np.abs(residuals)),
        '残差绝对值中位数': np.median(np.abs(residuals)),
        '正残差比例': np.mean(residuals > 0),
        '负残差比例': np.mean(residuals < 0),
        '残差绝对值<1比例': np.mean(np.abs(residuals) < 1),
        '残差绝对值<5比例': np.mean(np.abs(residuals) < 5),
        '残差绝对值>10比例': np.mean(np.abs(residuals) > 10)
    }

    return residuals, stats

def masked_loss(pred, target, mask, beta=2.5):
    """
    只在掩码为True的区域计算损失 (SmoothL1Loss)
    pred: [batch_size, 1, H, W]
    target: [batch_size, 1, H, W]
    mask: [batch_size, 1, H, W]
    beta: SmoothL1Loss参数
    """
    mask = mask.bool()
    # 只计算掩码区域的损失
    pred_masked = pred[mask]
    target_masked = target[mask]

    if len(pred_masked) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # 使用SmoothL1Loss
    criterion = nn.SmoothL1Loss(beta=beta)
    return criterion(pred_masked, target_masked)

def calculate_masked_metrics(y_true, y_pred, mask):
    """在掩码区域计算评估指标"""
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    if len(y_true_masked) == 0:
        return 0.0, 0.0, 0.0, 0.0

    mse = mean_squared_error(y_true_masked, y_pred_masked)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_masked, y_pred_masked)
    r2 = r2_score(y_true_masked, y_pred_masked)

    return mse, rmse, mae, r2


class GradientLoss(nn.Module):
    """
    Sobel 梯度损失 (Gradient Loss)
    计算图像在 X 和 Y 方向的梯度差异，强迫模型拟合边缘和形状。
    """

    def __init__(self, device):
        super(GradientLoss, self).__init__()
        # 定义 Sobel 算子 (水平和垂直方向)
        # 形状: [out_channels, in_channels, kH, kW] -> [1, 1, 3, 3]
        kernel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        kernel_y = torch.tensor([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 将卷积核移动到 GPU/CPU，并设为不可训练
        self.kernel_x = kernel_x.to(device)
        self.kernel_y = kernel_y.to(device)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, mask):
        """
        pred:   [batch, 1, H, W]
        target: [batch, 1, H, W]
        mask:   [batch, 1, H, W] (Boolean or Float 0/1)
        """
        # 如果 mask 全空，直接返回 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 确保 mask 是 float 类型
        mask_f = mask.float()

        # 计算预测值的梯度
        # padding=1 保证尺寸不变
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)

        # 计算真实值的梯度
        target_grad_x = F.conv2d(target, self.kernel_x, padding=1)
        target_grad_y = F.conv2d(target, self.kernel_y, padding=1)

        # 只在 mask 有效区域内计算梯度损失
        # 为什么要在 mask 区域计算？因为无数据区域的梯度是无意义的
        loss_x = self.l1(pred_grad_x * mask_f, target_grad_x * mask_f)
        loss_y = self.l1(pred_grad_y * mask_f, target_grad_y * mask_f)

        return loss_x + loss_y

class SmoothnessLoss(nn.Module):
    """
    平滑一致性损失 (Total Variation Loss)
    无需真值，仅约束预测图像在空间上的连续性，消除边界突变。
    """

    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, pred):
        # pred shape: (B, 1, H, W)
        # 计算水平方向梯度 (h_diff) 和 垂直方向梯度 (v_diff)
        h_diff = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        v_diff = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        # 计算平均变分
        loss = torch.mean(h_diff) + torch.mean(v_diff)
        return loss