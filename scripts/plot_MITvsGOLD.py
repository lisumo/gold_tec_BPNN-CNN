import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from ppgnss import gnss_utils
from matplotlib.colors import LinearSegmentedColormap

# 【新增】导入 GOLD 原始数据解析函数
import sys

project_root = "E:/research/TEC/2branches_BPNN+CNN"
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

# 只导入 process_ni1_file，时间解析用自己的函数
from src.data_process.gold_parser import process_ni1_file


# 【新增】自定义时间解析函数，专门处理 L1C 格式
def parse_l1c_time_from_path(file_path):
    """
    从 GOLD L1C 文件路径解析时间
    格式示例: GOLD_L1C_CHA_NI1_2023_293_00_10_v05_r01_c01.nc
    """
    file_name = os.path.basename(file_path)
    try:
        parts = file_name.split('_')

        # 找到 NI1 的位置，时间信息在其后: YYYY_DDD_HH_MM
        ni1_index = parts.index('NI1')
        year = int(parts[ni1_index + 1])
        julian_day = int(parts[ni1_index + 2])
        hour = int(parts[ni1_index + 3])
        minute = int(parts[ni1_index + 4])

        base_date = pd.Timestamp(f"{year}-01-01")
        target_date = base_date + pd.Timedelta(days=julian_day - 1)
        return pd.Timestamp(f"{target_date.date()} {hour:02d}:{minute:02d}:00")

    except (IndexError, ValueError, AttributeError) as e:
        raise ValueError(f"无法从L1C文件路径解析时间：{file_path}，错误：{str(e)}")


def load_mit_tec(target_time, lon_range, lat_range):
    """加载并提取MIT数据中目标时刻、目标经纬度范围的TEC数据"""
    print(f"正在加载MIT数据：{MIT_OBJ_PATH}")
    try:
        xr_mit = gnss_utils.loadobject(MIT_OBJ_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"MIT数据文件不存在：{MIT_OBJ_PATH}，请检查路径")
    except Exception as e:
        raise RuntimeError(f"加载MIT数据失败：{str(e)}")

    mit_time = target_time + np.timedelta64(150, "s")
    xr_mit_time = xr_mit.sel(time=mit_time, method="nearest")
    print(f"MIT数据实际匹配时刻：{pd.to_datetime(xr_mit_time.time.values)}")

    time_start = xr_mit.time[0]
    hour_f = (pd.to_datetime(xr_mit_time.time.values) - pd.to_datetime(time_start.values)) / np.timedelta64(1, "h")
    shift_mit = hour_f * 360 / 24 + 180
    shift_mit_int = int(np.round(shift_mit)) % xr_mit_time.sizes["lon"]
    xr_mit_roll = xr_mit_time.roll(lon=shift_mit_int, roll_coords=True).rename({"lon": "slon"})
    xr_mit_roll['slon'] = xr_mit_roll['slon'].where(
        xr_mit_roll['slon'] <= 180,
        xr_mit_roll['slon'] - 360
    )
    xr_mit_roll = xr_mit_roll.sortby('slon')

    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    xr_mit_tec = xr_mit_roll.sel(
        slon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max)
    )

    if xr_mit_tec.isnull().all():
        raise ValueError(f"MIT TEC数据在范围{lon_range}x{lat_range}内全为空，请检查范围或数据")
    print(f"MIT TEC数据维度：{xr_mit_tec.shape}，坐标范围：slon{lon_range}，lat{lat_range}")
    return xr_mit_tec


def load_gold_raw_data(nh_file_path, sh_file_path, lon_range, lat_range):
    """
    【关键修改】加载GOLD原始L1C数据（未经网格化处理）

    注意：process_ni1_file 返回的DataFrame使用中文列名：
    - '经度' (longitude)
    - '纬度' (latitude)
    - '辐射值' (radiance)

    参数:
        nh_file_path: 北半球原始文件路径 (CHA)
        sh_file_path: 南半球原始文件路径 (CHB)
        lon_range: 经度范围 (min, max)
        lat_range: 纬度范围 (min, max)

    返回:
        DataFrame包含原始像素数据，列名：'经度', '纬度', '辐射值'
    """
    print(f"\n正在加载GOLD原始数据...")
    print(f"北半球(CHA): {nh_file_path}")
    print(f"南半球(CHB): {sh_file_path}")

    try:
        # 解析原始数据，不进行网格化
        # process_ni1_file 返回 DataFrame，列名为：'经度', '纬度', '辐射值'
        nh_df, _ = process_ni1_file(nh_file_path, lon_range[0], lon_range[1], 0, lat_range[1])
        sh_df, _ = process_ni1_file(sh_file_path, lon_range[0], lon_range[1], lat_range[0], 0)

        # 合并南北半球数据
        all_df = pd.concat([nh_df, sh_df], ignore_index=True)

        if len(all_df) == 0:
            raise ValueError("GOLD原始数据在指定范围内为空")

        # 过滤指定范围外的数据（保险起见，虽然process_ni1_file已经筛选过）
        mask = (
                (all_df['经度'] >= lon_range[0]) &
                (all_df['经度'] <= lon_range[1]) &
                (all_df['纬度'] >= lat_range[0]) &
                (all_df['纬度'] <= lat_range[1])
        )
        all_df = all_df[mask].copy()

        print(f"GOLD原始数据点数量：{len(all_df)}")
        print(f"坐标范围：Lon {all_df['经度'].min():.1f}~{all_df['经度'].max():.1f}, "
              f"Lat {all_df['纬度'].min():.1f}~{all_df['纬度'].max():.1f}")
        print(f"辐射值范围：{all_df['辐射值'].min():.2f}~{all_df['辐射值'].max():.2f} Rayleighs")

        return all_df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"GOLD原始数据文件不存在：{str(e)}")
    except Exception as e:
        raise RuntimeError(f"加载GOLD原始数据失败：{str(e)}")


def add_map_features(ax):
    """添加地图底图要素（海岸线、国界）"""
    coastline = NaturalEarthFeature(
        category='physical', name='coastline', scale='10m',
        edgecolor='black', facecolor='none'
    )
    ax.add_feature(coastline, linewidth=0.3)

    borders = NaturalEarthFeature(
        category='cultural', name='admin_0_countries', scale='10m',
        edgecolor='black', facecolor='none'
    )
    ax.add_feature(borders, linewidth=0.2)


def add_gridlines(ax, lon_range, lat_range, show_left_labels=True):
    """添加简洁的虚线网格，通过参数控制是否显示左侧纬度标签"""
    import matplotlib.ticker as mticker

    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3,
                      color='gray', linewidth=0.5,
                      xlocs=np.arange(-90, -10, 10),
                      ylocs=np.arange(-40, 40, 10))

    gl.top_labels = False
    gl.right_labels = False
    # 【新增逻辑】根据传入的参数决定是否绘制左侧（纬度）标签
    gl.left_labels = show_left_labels

    gl.xlocator = mticker.FixedLocator([-80, -60, -40, -20])
    gl.ylocator = mticker.FixedLocator([-40, -30, -20, -10, 0, 10, 20, 30, 40])

    def lon_formatter(x, pos):
        if x < 0:
            return f'{int(abs(x))}°W'
        elif x > 0:
            return f'{int(x)}°E'
        else:
            return '0°'

    def lat_formatter(y, pos):
        if y < 0:
            return f'{int(abs(y))}°S'
        elif y > 0:
            return f'{int(y)}°N'
        else:
            return '0°'

    gl.xformatter = mticker.FuncFormatter(lon_formatter)
    gl.yformatter = mticker.FuncFormatter(lat_formatter)

    gl.xlabel_style = {'size': 10, 'family': 'Times New Roman'}
    gl.ylabel_style = {'size': 10, 'family': 'Times New Roman'}


def plot_mit_gold_comparison(gold_df_raw, xr_mit_tec, target_time, lon_range, lat_range):
    """绘制MIT TEC与GOLD原始辐射值对比图 - 科研绘图规范"""

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10
    })

    a4_width_inch = 6.27
    fig_height = 3.2

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(a4_width_inch, fig_height),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # 【关键修复1】重新分配水平空间：
    # wspace=0.45: 强制拉开两图间距，彻底容纳左侧的 Colorbar + Rayleighs/nm 标签
    # left=0.06, right=0.88: 整体左移，防止右侧 Colorbar 的 TECU 标签被图片右边缘切断
    plt.subplots_adjust(wspace=0.45, left=0.06, right=0.88, top=0.88, bottom=0.15)

    # ==================== 子图1: GOLD ====================
    scatter1 = ax1.scatter(
        gold_df_raw['经度'].values,
        gold_df_raw['纬度'].values,
        c=gold_df_raw['辐射值'].values,
        cmap="jet",
        vmin=0,
        vmax=PLOT_CONFIG["gold_vmax"],
        s=3,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        edgecolors='none'
    )

    add_map_features(ax1)
    add_gridlines(ax1, lon_range, lat_range, show_left_labels=True)

    ax1.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                   crs=ccrs.PlateCarree())

    # 【关键修复2】使用 set_title 将 (a) 稳定放置于图外左上角
    # loc='left' 保证左对齐，pad=8 控制与上边框的间距
    ax1.set_title('(a)', loc='left', fontsize=12, pad=8)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # 将 width 从 5% 微调至 4%，使其在单栏排版中显得不那么笨重
    cax1 = inset_axes(ax1, width="4%", height="85%", loc='center left',
                      bbox_to_anchor=(1.04, 0, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=1)
    cbar1 = fig.colorbar(scatter1, cax=cax1, orientation='vertical', extend='both')
    cbar1.set_label('Rayleighs(nm)', fontsize=10, rotation=90, labelpad=1)
    cbar1.ax.tick_params(labelsize=10)

    # ==================== 子图2: MIT TEC ====================
    slon_edges = np.linspace(
        xr_mit_tec.slon.min().values,
        xr_mit_tec.slon.max().values,
        len(xr_mit_tec.slon) + 1
    )
    lat_edges_mit = np.linspace(
        xr_mit_tec.lat.min().values,
        xr_mit_tec.lat.max().values,
        len(xr_mit_tec.lat) + 1
    )

    mesh2 = ax2.pcolormesh(
        slon_edges, lat_edges_mit, xr_mit_tec.values,
        cmap="jet",
        vmin=PLOT_CONFIG["mit_vmin"],
        vmax=PLOT_CONFIG["mit_vmax"],
        shading='flat', transform=ccrs.PlateCarree()
    )

    add_map_features(ax2)
    add_gridlines(ax2, lon_range, lat_range, show_left_labels=False)

    # 【关键修复2】右图同理，放置 (b)
    ax2.set_title('(b)', loc='left', fontsize=12, pad=8)

    cax2 = inset_axes(ax2, width="4%", height="85%", loc='center left',
                      bbox_to_anchor=(1.04, 0, 1, 1),
                      bbox_transform=ax2.transAxes, borderpad=1)
    cbar2 = fig.colorbar(mesh2, cax=cax2, orientation='vertical', extend='both')
    cbar2.set_label('TEC(TECU)', fontsize=10, rotation=90, labelpad=5)
    cbar2.ax.tick_params(labelsize=10)

    fig_name = f"MIT_GOLD_Compare_{target_time.strftime('%Y%m%d_%H%M')}.png"
    fig_path = os.path.join(OUTPUT_FIG_DIR, fig_name)
    plt.savefig(fig_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close()
    print(f"\n对比图已保存：{fig_path}")


def main():
    print("=" * 50)
    print(f"MIT-GOLD对比分析开始 | 目标时刻：{TARGET_TIME} | 范围：Lon{LON_RANGE} x Lat{LAT_RANGE}")
    print("【注意】GOLD子图现在展示的是原始L1C数据（未经网格化预处理）")
    print("=" * 50)

    try:
        # 加载MIT数据（保持原有逻辑）
        xr_mit_tec = load_mit_tec(TARGET_TIME, LON_RANGE, LAT_RANGE)

        # 【修改】加载GOLD原始数据（而非预处理后的网格数据）
        gold_df_raw = load_gold_raw_data(
            GOLD_NH_RAW_PATH,  # 北半球 CHA 文件路径【请填写】
            GOLD_SH_RAW_PATH,  # 南半球 CHB 文件路径【请填写】
            LON_RANGE,
            LAT_RANGE
        )

        plot_mit_gold_comparison(gold_df_raw, xr_mit_tec, TARGET_TIME, LON_RANGE, LAT_RANGE)

        print("\n" + "=" * 50)
        print("MIT-GOLD对比分析完成！")
        print("=" * 50)

    except Exception as e:
        print(f"\n分析失败：{str(e)}")
        raise


# -----------------------------------------------------------------------------------
# 【配置路径 - 请根据您的实际路径填写】
MIT_OBJ_PATH = "E:/research/TEC/mitg2023_2024.obj"

# 【关键修改】改为原始GOLD L1C文件路径（而非预处理后的网格文件）
# 注意：原始数据分为南北两个文件（CHA和CHB），需要分别指定
# 文件命名格式示例：GOLD_L1C_CHA_NI1_2023_293_00_10_v05_r01_c01.nc
GOLD_NH_RAW_PATH = "E:/research/TEC/2023/GOLD_L1C_CHA_NI1_2023_293_00_10_v05_r01_c01.nc"
GOLD_SH_RAW_PATH = "E:/research/TEC/2023/GOLD_L1C_CHB_NI1_2023_293_00_10_v05_r01_c02.nc"

OUTPUT_FIG_DIR = "MIT_GOLD_Compare"
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# 【修改】使用自定义的 L1C 时间解析函数
TARGET_TIME = parse_l1c_time_from_path(GOLD_NH_RAW_PATH)

LON_RANGE = (-90, -20)
LAT_RANGE = (-40, 30)

PLOT_CONFIG = {
    "gold_cmap": "jet",
    "mit_cmap": "jet",
    "gold_vmax": 1000,
    "mit_vmax": 150,
    "mit_vmin": 0,
    "dpi": 500,
}

if __name__ == "__main__":
    main()