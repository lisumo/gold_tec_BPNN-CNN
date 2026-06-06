import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """空间注意力模块，用于突出重要区域特征（添加缩放因子）"""

    def __init__(self, kernel_size=7, scale_factor=1.0):
        super(SpatialAttention, self).__init__()

        # 确保卷积核大小为奇数
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # 特征压缩与注意力权重生成
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 添加可学习的缩放因子，初始值为scale_factor
        self.scale = nn.Parameter(torch.tensor(scale_factor, dtype=torch.float32))

    def forward(self, x):
        # 沿通道维度计算最大值和平均值
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # 最大值池化
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # 平均值池化

        # 拼接两种池化结果
        x_cat = torch.cat([max_pool, avg_pool], dim=1)

        # 生成注意力权重图并应用缩放因子
        attention = self.conv1(x_cat)
        attention = self.sigmoid(attention)
        attention = attention * self.scale  # 应用缩放因子

        # 应用注意力权重
        return x * attention