# src/utils/options.py
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="TEC Fusion Model Server Training")

    # === 基础环境 ===
    parser.add_argument('--gpu_id', type=str, default='0', help='指定GPU ID')
    parser.add_argument('--exp_name', type=str, default='server_exp', help='实验名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # === 训练控制 ===
    parser.add_argument('--epochs', type=int, default=None, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--dataset_path', type=str, default=None, help='数据集路径')

    # === 模型模式 ===
    parser.add_argument('--model_mode', type=str, default=None, choices=['fusion', 'cnn', 'bpnn'], help='模型模式')
    parser.add_argument('--use_spatial_attention', type=int, default=None, help='是否使用空间注意力(0或1)')

    # === 关键修改：添加陆地遮挡控制参数 ===
    # 接收 0 或 1，后续转为 bool
    parser.add_argument('--land_masking', type=int, default=None, help='是否开启陆地遮挡实验 (0/1)')
    # 接收字符串格式: "lon_min,lon_max,lat_min,lat_max" 例如 "-70,-60,-30,-10"
    parser.add_argument('--mask_region', type=str, default=None, help='遮挡区域坐标 (lon_min,lon_max,lat_min,lat_max)')

    # === 高级功能 ===
    parser.add_argument('--resume', action='store_true', help='是否恢复训练')
    parser.add_argument('--log_interval', type=int, default=10, help='日志记录间隔')

    args = parser.parse_args()
    return args