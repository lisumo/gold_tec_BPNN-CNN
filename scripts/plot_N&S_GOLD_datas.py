import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==============================================================================
# 1. 环境与路径配置
# ==============================================================================
PROJECT_ROOT = "E:/research/TEC/2branches_BPNN+CNN"
if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入你本地定义的函数
# 请确保你的 gold_parser.py 里的函数名和参数与你提供的一致
from src.data_process.gold_parser import process_ni1_file

# 定义原始 L1C 数据文件路径
GOLD_NH_RAW_PATH = "E:/research/TEC/2023/GOLD_L1C_CHA_NI1_2023_293_23_40_v05_r01_c01.nc"
GOLD_SH_RAW_PATH = "E:/research/TEC/2023/GOLD_L1C_CHB_NI1_2023_293_23_40_v05_r01_c02.nc"

OUTPUT_IMAGE_PATH = "GOLD_Raw_Radiance_Unprocessed.png"


# ==============================================================================
# 2. 修改后的数据加载逻辑
# ==============================================================================
def load_raw_gold_data_fixed(nh_path, sh_path):
    # 设置一个较宽的范围，以观察高纬度和边缘的原始稀疏数据
    LON_MIN, LON_MAX = -180, 180
    LAT_MIN, LAT_MAX = -90, 90

    dfs = []

    # 处理北半球
    if os.path.exists(nh_path):
        print(f"Loading NH: {os.path.basename(nh_path)}")
        # 修正：传入5个参数，并解包返回的元组
        df_nh, wave_nh = process_ni1_file(nh_path, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        dfs.append(df_nh)

    # 处理南半球
    if os.path.exists(sh_path):
        print(f"Loading SH: {os.path.basename(sh_path)}")
        df_sh, wave_sh = process_ni1_file(sh_path, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        dfs.append(df_sh)

    if not dfs:
        raise ValueError("未能读取到任何数据，请检查路径。")

    return pd.concat(dfs, ignore_index=True)


# ==============================================================================
# 3. 绘图函数 (适配中文列名)
# ==============================================================================
def plot_radiance_sparsity(df, save_path):
    print("正在生成原始观测分布图...")

    # 适配你函数中定义的中文列名
    lons = df['经度'].values
    lats = df['纬度'].values
    rads = df['辐射值'].values

    fig = plt.figure(figsize=(4, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 添加底图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    # 重点：设置显示范围，包含高纬度地区
    ax.set_extent([-130, 20, -85, 85], crs=ccrs.PlateCarree())

    # 绘制原始点，s=1 甚至是更小的点能更清楚地看到“条纹”和“稀疏度”
    # 使用 cmap='viridis' 或 'magma' 观察亮度
    sc = ax.scatter(lons, lats, c=rads, s=0.8, cmap='viridis',
                    alpha=0.7, transform=ccrs.PlateCarree(),
                    vmin=0, vmax=np.percentile(rads, 98))  # 稍微压制极值，看清分布

    # 坐标轴和网格
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    plt.colorbar(sc, label='135.6 nm Radiance (Rayleighs)', shrink=0.6)
    plt.title('GOLD Raw Observation Density (High Latitude Sparsity Demo)', pad=20)

    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"绘图完成：{save_path}")
    plt.show()


if __name__ == "__main__":
    try:
        combined_df = load_raw_gold_data_fixed(GOLD_NH_RAW_PATH, GOLD_SH_RAW_PATH)
        plot_radiance_sparsity(combined_df, OUTPUT_IMAGE_PATH)
    except Exception as e:
        print(f"运行失败: {e}")