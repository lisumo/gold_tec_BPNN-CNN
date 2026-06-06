import numpy as np
import torch
import torch.nn as nn
from .components import SpatialAttention
from .backbones import BPNNBackgroundModel


class TECFusionCNNModel(nn.Module):
    """
    支持三种模式的通用模型:
    1. 'fusion': CNN + BPNN (加权融合)
    2. 'cnn': 仅 CNN (无背景场)
    3. 'bpnn': 仅 BPNN (仅背景场)
    """

    def __init__(self, params):
        super(TECFusionCNNModel, self).__init__()
        self.input_shape = params['input_shape']
        self.input_channels = params['input_channels']
        self.time_features_dim = params['time_features_dim']

        # 【新增】读取模式
        self.model_mode = params.get('model_mode', 'fusion')
        print(f"Build Model Mode: {self.model_mode}")

        self.use_spatial_attention = params['use_spatial_attention']
        self.attention_kernel_size = params['attention_kernel_size']
        self.attention_scale = params.get('attention_scale', 1.0)
        self.fusion_weight = params.get('fusion_weight', 0.5)

        # ================== 构建 CNN 部分 (Fusion 或 CNN 模式) ==================
        if self.model_mode in ['fusion', 'cnn']:
            # 空间分支 (CNN)
            self.spatial_branch = self._build_conv_branch(
                input_channels=self.input_channels,
                conv_channels=params['conv_channels'],
                kernel_sizes=params['kernel_sizes']
            )

            # 计算平坦化尺寸
            self.spatial_flat_size = self._get_branch_flat_size(
                self.input_shape, self.spatial_branch, input_channels=self.input_channels
            )

            # 空间全连接
            spatial_dense_blocks = []
            spatial_layers = [self.spatial_flat_size] + params['spatial_dense_layers']
            for i in range(1, len(spatial_layers)):
                spatial_dense_blocks.append(nn.Linear(spatial_layers[i - 1], spatial_layers[i]))
                spatial_dense_blocks.append(nn.LeakyReLU(negative_slope=0.01))
                if params['dropout_rate'] > 0:
                    spatial_dense_blocks.append(nn.Dropout(params['dropout_rate']))
            self.spatial_dense = nn.Sequential(*spatial_dense_blocks)
            self.spatial_output_size = spatial_layers[-1]

            # 时间分支 (配合CNN)
            time_dense_blocks = []
            time_layers = [self.time_features_dim] + params['time_dense_layers']
            for i in range(1, len(time_layers)):
                time_dense_blocks.append(nn.Linear(time_layers[i - 1], time_layers[i]))
                time_dense_blocks.append(nn.LeakyReLU(negative_slope=0.01))
                if params['dropout_rate'] > 0:
                    time_dense_blocks.append(nn.Dropout(params['dropout_rate']))
            self.time_branch = nn.Sequential(*time_dense_blocks)
            self.time_output_size = time_layers[-1]

            # 融合层 (CNN内部的空间+时间融合)
            fusion_input_size = self.spatial_output_size + self.time_output_size
            fusion_layers = [fusion_input_size] + params['fusion_dense_layers']
            fusion_blocks = []
            for i in range(1, len(fusion_layers)):
                fusion_blocks.append(nn.Linear(fusion_layers[i - 1], fusion_layers[i]))
                fusion_blocks.append(nn.LeakyReLU(negative_slope=0.01))
                if params['dropout_rate'] > 0:
                    fusion_blocks.append(nn.Dropout(params['dropout_rate']))

            # CNN 最终输出层
            fusion_blocks.append(nn.Linear(fusion_layers[-1], np.prod(self.input_shape)))
            self.fusion_dense = nn.Sequential(*fusion_blocks)
            self.output_shape = self.input_shape

        # ================== 构建 BPNN 部分 (Fusion 或 BPNN 模式) ==================
        if self.model_mode in ['fusion', 'bpnn']:
            self.bpnn_model = BPNNBackgroundModel(params)

            # 只有融合模式才需要可学习的权重
            if self.model_mode == 'fusion':
                self.fusion_alpha = nn.Parameter(torch.tensor(self.fusion_weight, dtype=torch.float32))


    def _build_conv_branch(self, input_channels, conv_channels, kernel_sizes):
        """构建卷积分支"""
        layers = []
        in_channels = input_channels

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels 与 kernel_sizes 长度不匹配")

        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            ))
            layers.append(nn.LeakyReLU(negative_slope=0.01))

            if self.use_spatial_attention and (i == len(conv_channels) - 1 or i % 2 == 0):
                layers.append(SpatialAttention(
                    kernel_size=self.attention_kernel_size,
                    scale_factor=self.attention_scale
                ))

            in_channels = out_channels

        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def _get_branch_flat_size(self, input_shape, branch, input_channels):
        with torch.no_grad():
            x = torch.zeros(1, input_channels, *input_shape)
            x = branch(x)
            return x.view(1, -1).size(1)

    def forward(self, spatial_x, time_x):
        batch_size = spatial_x.size(0)
        cnn_output = None
        bpnn_output = None
        final_output = None

        # 1. 计算 CNN 输出
        if self.model_mode in ['fusion', 'cnn']:
            spatial_features = self.spatial_branch(spatial_x)
            spatial_features = self.spatial_dense(spatial_features)
            time_features = self.time_branch(time_x)
            fused_features = torch.cat([spatial_features, time_features], dim=1)
            cnn_output = self.fusion_dense(fused_features)
            cnn_output = cnn_output.view(-1, 1, *self.output_shape)

        # 2. 计算 BPNN 输出
        if self.model_mode in ['fusion', 'bpnn']:
            bpnn_output = self.bpnn_model(time_x, spatial_x, batch_size)

        # 3. 根据模式返回
        if self.model_mode == 'fusion':
            alpha = torch.sigmoid(self.fusion_alpha)
            final_output = alpha * cnn_output + (1 - alpha) * bpnn_output
            return final_output, cnn_output, bpnn_output

        elif self.model_mode == 'cnn':
            # 仅 CNN 模式，final 就是 cnn
            return cnn_output, cnn_output, None

        elif self.model_mode == 'bpnn':
            # 仅 BPNN 模式，final 就是 bpnn
            return bpnn_output, None, bpnn_output