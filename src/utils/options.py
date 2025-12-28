# src/utils/options.py
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="TEC Fusion Model Server Training")

    # === 基础环境 ===
    parser.add_argument('--gpu_id', type=str, default='0', help='指定GPU ID, 如 "0" 或 "0,1"')
    parser.add_argument('--exp_name', type=str, default='server_exp', help='实验名称，用于生成日志文件夹')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # === 训练控制 ===
    parser.add_argument('--epochs', type=int, default=None, help='训练轮次 (覆盖Config)')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小 (覆盖Config)')
    parser.add_argument('--lr', type=float, default=None, help='学习率 (覆盖Config)')

    # === 路径控制 ===
    # 允许服务器使用不同的数据路径，如果不传则使用 Config 中的默认路径
    parser.add_argument('--dataset_path', type=str, default=None, help='数据集.pth路径')

    # === 高级功能 ===
    parser.add_argument('--resume', action='store_true', help='是否尝试从该实验的 last_checkpoint 恢复训练')
    parser.add_argument('--log_interval', type=int, default=10, help='多少个epoch记录一次日志')

    args = parser.parse_args()
    return args