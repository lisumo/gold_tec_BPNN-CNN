import torch
import torch.nn as nn

class BPNNBackgroundModel(nn.Module):
    """BP神经网络模型，用于生成大尺度TEC背景场"""

    def __init__(self, params):
        super(BPNNBackgroundModel, self).__init__()

        self.input_shape = params['input_shape']  # (height, width)
        self.time_features_dim = params['time_features_dim']  # 时间特征维度
        self.bpnn_hidden_layers = params['bpnn_hidden_layers']  # BPNN隐藏层配置
        self.dropout_rate = params['dropout_rate']

        # 输入特征：时间特征 + 空间位置编码 + 全部12个空间特征通道
        # 空间位置编码：每个网格点的经纬度标准化坐标
        # 空间特征：全部12个通道
        self.register_buffer('spatial_pos_encoding', self._create_spatial_position_encoding())
        spatial_pos_dim = 2  # 经度和纬度两个维度
        spatial_features_dim = 12  # 全部12个空间特征通道

        # 总输入维度 = 时间特征 + 空间位置编码 + 空间特征
        input_dim = self.time_features_dim + spatial_pos_dim + spatial_features_dim

        # 构建BPNN网络
        layers = []
        layers_list = [input_dim] + self.bpnn_hidden_layers

        for i in range(len(layers_list) - 1):
            layers.append(nn.Linear(layers_list[i], layers_list[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

        # 输出层：预测单个TEC值
        layers.append(nn.Linear(layers_list[-1], 1))

        self.network = nn.Sequential(*layers)

    def _create_spatial_position_encoding(self):
        """创建空间位置编码（标准化经纬度坐标）"""
        height, width = self.input_shape
        # 生成网格坐标
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        # 形状: (2, height, width)
        pos_encoding = torch.stack([x_coords, y_coords], dim=0)
        return pos_encoding

    def forward(self, time_features, spatial_features, batch_size):
        """
        前向传播
        Args:
            time_features: 时间特征 [batch_size, time_features_dim]
            spatial_features: 空间特征 [batch_size, channels, height, width]
            batch_size: 批次大小
        Returns:
            background: 背景TEC场 [batch_size, 1, height, width]
        """
        # 扩展空间位置编码到批次维度
        spatial_pos = self.spatial_pos_encoding.unsqueeze(0)  # [1, 2, H, W]
        spatial_pos = spatial_pos.repeat(batch_size, 1, 1, 1)  # [batch_size, 2, H, W]

        # 重塑为 [batch_size * H * W, 2]
        batch_size, _, height, width = spatial_pos.shape
        spatial_pos_flat = spatial_pos.permute(0, 2, 3, 1).reshape(-1, 2)

        # 扩展时间特征到每个空间位置 [batch_size * H * W, time_features_dim]
        time_features_expanded = time_features.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, time_features_dim]
        time_features_expanded = time_features_expanded.repeat(1, height, width, 1)  # [batch_size, H, W, time_features_dim]
        time_features_flat = time_features_expanded.reshape(-1, self.time_features_dim)

        # 处理全部12个空间特征通道 [batch_size * H * W, 12]
        spatial_flat = spatial_features.permute(0, 2, 3, 1).reshape(-1, 12)

        # 拼接所有特征 [batch_size * H * W, time_features_dim + 2 + 12]
        combined_features = torch.cat([time_features_flat, spatial_pos_flat, spatial_flat], dim=1)

        # 通过BPNN网络
        output_flat = self.network(combined_features)  # [batch_size * H * W, 1]

        # 重塑为空间维度 [batch_size, 1, H, W]
        background = output_flat.reshape(batch_size, height, width, 1).permute(0, 3, 1, 2)

        return background