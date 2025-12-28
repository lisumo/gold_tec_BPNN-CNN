# src/utils/common.py
import torch

def setup_gpu():
    """检查GPU是否可用，并配置设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nGPU：{torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        return device
    else:
        device = torch.device("cpu")
        print("\n未检测到可用GPU，将使用CPU")
        return device