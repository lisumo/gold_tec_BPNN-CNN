import sys
import os
import numpy as np
import pandas as pd
import xarray as xr

# 添加项目根目录到路径，以便读取 Config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config

# 尝试导入 ppgnss，如果环境没配好可能会报错
try:
    from ppgnss import gnss_utils
except ImportError:
    print("错误: 未找到 ppgnss 库，请确保在正确的环境中运行")
    sys.exit(1)


def inspect_mit_file():
    # 1. 获取文件路径
    mit_path = Config.MIT_FILE_PATH
    print(f"📂 正在检查 MIT 数据文件: {mit_path}")

    if not os.path.exists(mit_path):
        print("❌ 错误: 文件不存在，请检查 config.py 中的路径配置")
        return

    # 2. 加载数据
    try:
        # 使用项目原本的加载方式
        mit_data = gnss_utils.loadobject(mit_path)
        print("✅ 加载成功！")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 3. 检查是否有时间维度
    if not hasattr(mit_data, 'time'):
        print("⚠️ 警告: 该数据对象中没有发现 'time' 维度，请检查数据结构。")
        print("可用维度/变量:", mit_data.keys() if hasattr(mit_data, 'keys') else "未知")
        return

    # 4. 分析时间序列
    times = mit_data.time.values
    # 转换为 pandas datetime 以便格式化
    times_pd = pd.to_datetime(times)

    total_points = len(times)
    print(f"\n📊 === 时间维度统计 ===")
    print(f"总时间点数: {total_points}")
    if total_points > 0:
        print(f"开始时间: {times_pd[0]}")
        print(f"结束时间: {times_pd[-1]}")

    # 5. 计算时间分辨率 (Diff)
    if total_points > 1:
        # 计算相邻时间点的差值
        diffs = np.diff(times)
        # 转换为分钟
        diffs_min = diffs.astype('timedelta64[m]').astype(int)

        # 统计所有出现的间隔
        unique_diffs, counts = np.unique(diffs_min, return_counts=True)

        print(f"\n⏱️ === 时间分辨率 (间隔) ===")
        for diff, count in zip(unique_diffs, counts):
            print(f"  - 间隔 {diff} 分钟: 出现 {count} 次")

        # 判断是否均匀
        if len(unique_diffs) == 1:
            print(f"结论: 数据是均匀分布的，分辨率为 {unique_diffs[0]} 分钟")
        else:
            print(f"结论: 数据分布不均匀，存在多种时间间隔！")

    # 6. 打印详细样本 (用于排查 23:02:30 这种奇怪的时间点)
    print(f"\n🔍 === 时间点采样 (前20个) ===")
    for t in times_pd[:20]:
        print(f"  {t}")

    print(f"\n🔍 === 时间点采样 (中间部分) ===")
    mid = total_points // 2
    for t in times_pd[mid:mid + 10]:
        print(f"  {t}")


if __name__ == "__main__":
    inspect_mit_file()