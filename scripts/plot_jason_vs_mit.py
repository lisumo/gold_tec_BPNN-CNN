import os

# 1. 环境变量配置 (必须)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib

matplotlib.use('Agg')

import sys
import shutil
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.data_process.mit_parser import load_mit_tec
from src.data_process.jason_parser import find_jason_files_for_timestamp

try:
    import cartopy.crs as ccrs
    from cartopy.feature import NaturalEarthFeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️ 警告: 未检测到 Cartopy")

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Times New Rome']
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.font_manager as fm

fm.fontManager.addfont('/home/ymli/Fonts/times.ttf')

# 后续的 rcParams 配置保持不变即可生效
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12
})

# =================================================================
# Worker 函数 (保持不变)
# =================================================================
def process_single_timestamp(target_time, jason_dir, extent):
    import os
    import shutil
    import uuid
    import xarray as xr
    import pandas as pd
    import numpy as np
    from datetime import timedelta

    files = find_jason_files_for_timestamp(jason_dir, target_time, tolerance_minutes=60)
    if not files: return None

    lon_range = (extent[0], extent[1])
    lat_range = (extent[2], extent[3])

    dfs = []
    for f_path in files:
        temp_path = None
        try:
            unique_name = f"jason_{uuid.uuid4().hex}.nc"
            temp_path = os.path.join("/tmp", unique_name)
            shutil.copy(f_path, temp_path)

            ds_base = xr.open_dataset(temp_path, group='data_01', engine='h5netcdf')
            ds_ku = xr.open_dataset(temp_path, group='data_01/ku', engine='h5netcdf')

            ds_base.load();
            ds_ku.load()

            time = pd.to_datetime(ds_base['time'].values)
            lat = ds_base['latitude'].values
            lon = ds_base['longitude'].values
            surf = ds_base['rad_surface_type_flag'].values if 'rad_surface_type_flag' in ds_base else ds_base[
                'surface_type'].values

            if 'iono_cor_alt_ku' in ds_ku:
                iono = ds_ku['iono_cor_alt_ku'].values
            elif 'iono_cor_alt' in ds_ku:
                iono = ds_ku['iono_cor_alt'].values
            else:
                ds_base.close();
                ds_ku.close();
                continue

            ds_base.close();
            ds_ku.close()

            lon = np.where(lon > 180, lon - 360, lon)
            df = pd.DataFrame({'time': time, 'lat': lat, 'lon': lon, 'iono_cor': iono, 'surface': surf})

            df = df[(df['time'] - target_time).abs() <= timedelta(minutes=300)]
            df = df[(df['lon'] >= lon_range[0]) & (df['lon'] <= lon_range[1]) &
                    (df['lat'] >= lat_range[0]) & (df['lat'] <= lat_range[1])]
            df = df[df['surface'] == 0]
            df = df.dropna(subset=['iono_cor'])

            if len(df) > 0:
                dfs.append(df)
        except Exception:
            continue
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    if dfs: return pd.concat(dfs, ignore_index=True)
    return None


def plot_data_distribution():
    # 配置
    start_date = pd.to_datetime("2023-01-01")
    duration_days = 10
    jason_dir = "/mnt/public/Jason3_GDR"
    save_dir = os.path.join(project_root, 'logs', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "data_distribution_final.png")
    extent = [Config.MASK_LON_MIN, Config.MASK_LON_MAX, Config.MASK_LAT_MIN, Config.MASK_LAT_MAX]

    print(f"🚀 开始绘制最终版数据分布图")
    print(f"📅 预定观测时段: {start_date.strftime('%Y-%m-%d')} 起 {duration_days} 天")

    # 1. 加载 MIT
    print("\n[1/3] 加载 MIT GNSS 数据...")
    try:
        xr_mit = load_mit_tec(
            start_date + timedelta(hours=12),
            (extent[0], extent[1]), (extent[2], extent[3]),
            Config.MIT_FILE_PATH, time_tolerance=timedelta(minutes=30)
        )
        if xr_mit is not None:
            mit_lon, mit_lat, mit_tec = xr_mit.slon.values, xr_mit.lat.values, xr_mit.values
            mit_mask = ~np.isnan(mit_tec) & (mit_tec > 0)
            mit_tec_masked = np.ma.masked_where(~mit_mask, mit_tec)
            print("   -> MIT 加载成功")
        else:
            mit_tec_masked = None
    except Exception:
        mit_tec_masked = None

    # 2. 加载 Jason
    print(f"\n[2/3] 扫描 Jason-3 轨道 (启动多进程)...")
    check_times = [start_date + timedelta(hours=h) for h in range(0, duration_days * 24, 2)]
    jason_results = []
    num_workers = min(10, os.cpu_count() or 1)

    with mp.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        func = partial(process_single_timestamp, jason_dir=jason_dir, extent=extent)
        results = pool.map(func, check_times)
        for res in results:
            if res is not None and not res.empty:
                jason_results.append(res)
            sys.stdout.write(f"\r   -> 已处理任务: {len(jason_results)} / {len(check_times)}")
            sys.stdout.flush()

    print(f"\n   -> 扫描完成。")
    if jason_results:
        df_jason_all = pd.concat(jason_results, ignore_index=True)

        # 【新增】将电离层改正(m)转换为 TEC (TECU)
        # Jason-3 Ku-band (13.575 GHz) 转换系数约为 457.27
        df_jason_all['tec'] = -df_jason_all['iono_cor'] * 457.27

        vmin_jason = df_jason_all['tec'].quantile(0.01)
        vmax_jason = df_jason_all['tec'].quantile(0.99)

        # 【新增】打印实际时间范围
        actual_start = df_jason_all['time'].min()
        actual_end = df_jason_all['time'].max()
        print(f"📊 图中数据实际时间范围: {actual_start} 至 {actual_end}")
    else:
        df_jason_all = pd.DataFrame()

# 3. 绘图 (科研绘图规范版)
    print("\n[3/3] 生成地图...")
    if HAS_CARTOPY:
        # 【修改点1】全局强行覆盖为 Times New Roman 和 12 号字
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 12,
            'legend.title_fontsize': 12
        })

        # 【修改点2】单栏宽度约 6.27 英寸。设置高度为 5.2 英寸。
        # 由于投影会锁定长宽比为正方形，Matplotlib 会自动将图居中，左右自动留白。
        a4_width_inch = 6.27
        fig_height = 5.2

        fig = plt.figure(figsize=(a4_width_inch, fig_height))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # --- A. 地图线条 ---
        coast = NaturalEarthFeature('physical', 'coastline', '50m',
                                    edgecolor='black', facecolor='none',
                                    linewidth=0.6, zorder=1)
        borders = NaturalEarthFeature('cultural', 'admin_0_countries', '50m',
                                      edgecolor='gray', facecolor='none',
                                      linestyle=':', linewidth=0.4, zorder=1)
        ax.add_feature(coast)
        ax.add_feature(borders)

        # --- B. MIT 覆盖 ---
        if mit_tec_masked is not None:
            mesh = ax.pcolormesh(mit_lon, mit_lat, mit_tec_masked,
                                 cmap=mcolors.ListedColormap(['#A0C8F0']),
                                 vmin=0, vmax=1, alpha=0.8,
                                 transform=ccrs.PlateCarree(),
                                 zorder=2)

        # --- C. Jason 轨迹 ---
        if len(df_jason_all) > 0:
            cmap_jason = 'winter'
            sc = ax.scatter(df_jason_all['lon'], df_jason_all['lat'],
                            c=df_jason_all['tec'],
                            cmap=cmap_jason,
                            vmin=vmin_jason, vmax=vmax_jason,
                            s=0.1, alpha=0.3, marker='.',
                            transform=ccrs.PlateCarree(),
                            zorder=3)

            cb = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.03, aspect=15, shrink=0.85)
            cb.set_alpha(1.0)
            cb._draw_all()
            # 【修改点3】Colorbar 标签统一字体
            cb.set_label('TEC (TECU)', fontsize=12, family='Times New Roman')
            cb.ax.tick_params(labelsize=12)

        # 范围与装饰
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # 【修改点4】图例 12号字体，并放大图例标记以匹配大字号
        import matplotlib.patches as mpatches
        legend_patches = [
            mpatches.Patch(color='#A0C8F0', alpha=0.8, label='Land GNSS (MIT)'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=plt.get_cmap('winter')(0.8) if len(df_jason_all) > 0 else 'gray',
                       markersize=8, label='Jason-3 (Ocean)')
        ]
        leg = ax.legend(handles=legend_patches, loc='lower left',
                        framealpha=0.9, fontsize=12, facecolor='white')
        leg.set_title("Data Source", prop={'size': 12, 'family': 'Times New Roman'})

        # 【修改点5】网格线刻度统一字体
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4,
                          color='gray', linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12, 'family': 'Times New Roman'}
        gl.ylabel_style = {'size': 12, 'family': 'Times New Roman'}

        # 【核心修改点】使用 tight_layout 优化内部间距，但坚决不在 savefig 中使用 bbox_inches='tight'。
        # 强制图片文件背景为白底并保持 6.27 英寸的绝对物理宽度。
        fig.tight_layout()
        plt.savefig(save_path, dpi=300, facecolor='white')
        plt.close()
        print(f"✅ 最终优化图片已保存至: {save_path}")
    else:
        print("缺少 Cartopy，无法绘图")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    plot_data_distribution()