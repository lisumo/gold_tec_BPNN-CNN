import sys
import os
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.dataset.loader import GoldMit2DDataset

def main():
    print("=== 开始构建数据集 (2023-2024 跨年版) ===")

    try:
        # 1. 确定数据源路径
        # 优先使用 Config.GOLD_PATHS (列表)，如果不存在则回退到 OUTPUT_DIR_GOLD
        if hasattr(Config, 'GOLD_PATHS'):
            gold_source = Config.GOLD_PATHS
        else:
            gold_source = Config.OUTPUT_DIR_GOLD

        print(f"使用 GOLD 数据源: {gold_source}")
        print(f"使用 MIT 数据源: {Config.OUTPUT_DIR_MIT}")

        # 2. 初始化数据集
        # loader 已经更新为支持列表输入
        dataset = GoldMit2DDataset(
            gold_dir=gold_source,
            mit_dir=Config.OUTPUT_DIR_MIT,
            time_tolerance_minutes=Config.TIME_TOLERANCE_MINUTES,
            use_interpolation=Config.USE_INTERPOLATION
        )

        print(f"数据集构建完成，包含 {len(dataset)} 个样本")

        # 3. 保存逻辑 (周期性划分)
        if Config.USE_PERIODIC_SPLIT:
            print(f"\n🔀 [Mode] 启用周期性时间窗口划分 (Window={Config.PERIODIC_SPLIT_PARAMS['window_size']} days)")

            # 执行划分 (Config.START_DOY 此处仅作为占位符，loader 内部会使用 min_date 自动计算)
            train_ds, val_ds, test_ds = dataset.split_by_period(
                start_doy=Config.START_DOY,
                window_size=Config.PERIODIC_SPLIT_PARAMS['window_size'],
                train_days=Config.PERIODIC_SPLIT_PARAMS['train_days'],
                val_days=Config.PERIODIC_SPLIT_PARAMS['val_days']
            )

            # 构造文件名
            base_name, ext = os.path.splitext(Config.DATASET_PATH)

            # 分别保存
            train_ds.save_dataset(f"{base_name}_train{ext}")
            val_ds.save_dataset(f"{base_name}_val{ext}")
            test_ds.save_dataset(f"{base_name}_test{ext}")

            print(f"✅ 已保存划分数据集:\n - {base_name}_train{ext}\n - {base_name}_val{ext}\n - {base_name}_test{ext}")

            # 验证用：加载训练集
            check_path = f"{base_name}_train{ext}"
        else:
            print("\n⏹ [Mode] 默认模式（随机划分，保存全量文件）")
            dataset.save_dataset(Config.DATASET_PATH)
            check_path = Config.DATASET_PATH

        # === 验证环节 ===
        print(f"\n=== 验证加载: {os.path.basename(check_path)} ===")
        if os.path.exists(check_path):
            loaded_dataset = GoldMit2DDataset.load_dataset(check_path)
            dataloader = DataLoader(loaded_dataset, batch_size=4, shuffle=True)

            for batch in dataloader:
                # 兼容 6 或 7 个元素的解包 (dataset 现在返回 7 个元素，包含 time_str)
                if len(batch) >= 6:
                    print(f"空间特征: {batch[0].shape}")
                    print(f"目标数据: {batch[2].shape}")
                    if len(batch) > 6:
                        print(f"时间标签示例: {batch[6][0]}")
                break
        else:
            print("文件不存在，跳过验证。")

    except Exception as e:
        print(f"程序错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()