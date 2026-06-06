import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score

# 强制无GUI渲染
import matplotlib

matplotlib.use('Agg')

try:
    import cartopy.crs as ccrs
    from cartopy.feature import NaturalEarthFeature

    HAS_CARTOPY = True
except ImportError:
    raise ImportError("Cartopy required.")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.models.fusion import TECFusionCNNModel

import matplotlib.font_manager as fm

fm.fontManager.addfont('/home/ymli/Fonts/times.ttf')

# 全局排版定调
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# =================================================================
# ⚙️ 2x3 Jason-3 匹配与叠加验证组图配置区
# =================================================================
TARGET_TIME_STR = "20231216_232200"

DATASET_PATH_OVERRIDE = "/mnt/public/GOLD/2branches_BPNN+CNN/dataset/tec_2d_enhanced_dataset_test.pth"
MODEL_PATH = "/mnt/public/GOLD/2branches_BPNN+CNN/logs/exp1_fusion_full/best_model.pth"
JASON_CSV_PATH = "/mnt/public/GOLD/2branches_BPNN+CNN/logs/jason_comparison_results/jason_binned_stats_multi.csv"

OUTPUT_DIR = os.path.join(project_root, 'logs', 'paper_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAMES = ["BPNN", "CNN", "Fusion"]
ABC_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def add_map_outline(ax):
    coast = NaturalEarthFeature('physical', 'coastline', '50m',
                                edgecolor='black', facecolor='none', linewidth=0.4, zorder=3)
    ax.add_feature(coast)


def format_gridlines(ax, show_lon=True, show_lat=True):
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3, color='gray', linewidth=0.3, zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_lat
    gl.bottom_labels = show_lon

    gl.xlocator = mticker.FixedLocator([-80, -60, -40, -20])
    gl.ylocator = mticker.FixedLocator([-30, -10, 10, 30])

    def lon_fmt(x, pos): return f'{int(abs(x))}°W' if x < 0 else f'{int(x)}°E' if x > 0 else '0°'

    def lat_fmt(y, pos): return f'{int(abs(y))}°S' if y < 0 else f'{int(y)}°N' if y > 0 else '0°'

    gl.xformatter = mticker.FuncFormatter(lon_fmt)
    gl.yformatter = mticker.FuncFormatter(lat_fmt)
    gl.xlabel_style = {'size': 8, 'family': 'Times New Roman'}
    gl.ylabel_style = {'size': 8, 'family': 'Times New Roman'}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Phase 1/2] 载入底层地图与 {TARGET_TIME_STR} Jason-3匹配点...")

    # -----------------------------------------------------------
    # 1. 重建海洋底图场
    # -----------------------------------------------------------
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = checkpoint['config']
    scaler_y = checkpoint['scaler_y']
    scaler_X_spatial = checkpoint['scaler_X_spatial']
    scaler_X_time = checkpoint['scaler_X_time']

    model = TECFusionCNNModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    samples = torch.load(DATASET_PATH_OVERRIDE, weights_only=False)

    target_ts = pd.to_datetime(TARGET_TIME_STR, format="%Y%m%d_%H%M%S")
    target_sample = None
    min_diff = pd.Timedelta(days=100)

    for s in samples:
        time_label = s[6] if len(s) > 6 else ""
        try:
            if "_" in str(time_label):
                curr_ts = pd.to_datetime(str(time_label), format="%Y%m%d_%H%M%S")
            else:
                curr_ts = pd.to_datetime(str(time_label))
            diff = abs(curr_ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                target_sample = s
        except Exception:
            continue

    if target_sample is None:
        raise ValueError("底图数据池已空，提取失败。")

    map_data_pool = {}
    with torch.no_grad():
        sp = target_sample[0].numpy()
        C, H, W = sp.shape
        sp_norm = scaler_X_spatial.transform(sp.reshape(1, -1)).reshape(C, H, W)
        tm = target_sample[1].numpy().reshape(1, -1)
        tm_norm = scaler_X_time.transform(tm).reshape(-1)

        b_sp = torch.FloatTensor(sp_norm).unsqueeze(0).to(device)
        b_tm = torch.FloatTensor(tm_norm).unsqueeze(0).to(device)

        final_out, cnn_out, bpnn_out = model(b_sp, b_tm)

        def inverse_to_map(tensor):
            arr = tensor.cpu().numpy().reshape(1, -1)
            return scaler_y.inverse_transform(arr).reshape(config['input_shape'])

        map_data_pool['Fusion'] = inverse_to_map(final_out)
        map_data_pool['CNN'] = inverse_to_map(cnn_out)
        map_data_pool['BPNN'] = inverse_to_map(bpnn_out)
        mask_gold = target_sample[5].numpy().reshape(config['input_shape']).astype(bool)

    # -----------------------------------------------------------
    # 2. 从真实 CSV 文件加载与验证点匹配数据
    # -----------------------------------------------------------
    if not os.path.exists(JASON_CSV_PATH):
        raise FileNotFoundError(f"未找到已匹配的 Jason 验证 CSV 文件：{JASON_CSV_PATH}")

    jason_df = pd.read_csv(JASON_CSV_PATH)
    jason_df['model_time'] = jason_df['model_time'].astype(str)

    # 首先尝试精确字符串匹配
    target_df = jason_df[jason_df['model_time'] == TARGET_TIME_STR]
    target_scat_df = jason_df.copy()

    # 智能回退防错：基于时间差寻找绝对最接近的片段
    if len(target_df) == 0:
        jason_df['time_dt'] = pd.to_datetime(jason_df['model_time'], format="%Y%m%d_%H%M%S", errors='coerce')
        if jason_df['time_dt'].isnull().all():
            jason_df['time_dt'] = pd.to_datetime(jason_df['model_time'], errors='coerce')

        jason_df['time_diff'] = abs(jason_df['time_dt'] - target_ts)
        closest_time = jason_df.loc[jason_df['time_diff'].idxmin(), 'model_time']
        print(f"⚠️ CSV 内未匹配到精确字符串，自动寻获并使用最接近时段：{closest_time}")
        target_df = jason_df[jason_df['model_time'] == closest_time]

    print(f"📊 成功加载 {len(target_df)} 个 Jason-3 交叉验证数据点。")

    scatter_x_pool = {}
    scatter_y_pool = {}
    jason_lats = target_df['lat'].values
    jason_lons = target_df['lon'].values
    jason_true_tec = target_df['jason_tec'].values
    jason_true_tec_scat = target_scat_df['jason_tec'].values

    # 统筹背景图与采样点全集的物理极值，保证制图严谨对称
    all_map_values = []
    for name in MODEL_NAMES:
        valid_pvs = map_data_pool[name][mask_gold]
        all_map_values.extend(valid_pvs[~np.isnan(valid_pvs)])

        # 兼容匹配 CSV 中的模型预测列名
        if name in target_scat_df.columns:
            scatter_x_pool[name] = jason_true_tec_scat
            scatter_y_pool[name] = target_scat_df[name].values
        else:
            match_col = [col for col in target_scat_df.columns if name.lower() in col.lower()][0]
            scatter_x_pool[name] = jason_true_tec_scat
            scatter_y_pool[name] = target_scat_df[match_col].values

    global_vmin = 0.0
    global_vmax = max(float(np.max(all_map_values)), float(np.max(jason_true_tec))) if all_map_values else 80.0

    # =================================================================
    # [Phase 2/2] Jason-3 散点与底图叠加 (2x3 Matrix)
    # =================================================================
    print("\n🎨 [Phase 2/2] 开始渲染：启动散点核密度计算及背景小方块悬浮叠加...")

    map_shape = mask_gold.shape
    lon_arr = np.linspace(Config.MASK_LON_MIN, Config.MASK_LON_MAX, map_shape[1])
    lat_arr = np.linspace(Config.MASK_LAT_MIN, Config.MASK_LAT_MAX, map_shape[0])
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    a4_width_inch = 6.27
    fig_height = 4.4
    fig = plt.figure(figsize=(a4_width_inch, fig_height))

    from matplotlib.gridspec import GridSpec
    gs_master = GridSpec(1, 2, figure=fig, width_ratios=[5.67, 0.60], wspace=0.10)

    wspace_val = 0.20
    hspace_val = 0.35

    gs_left = gs_master[0].subgridspec(2, 3, wspace=wspace_val, hspace=hspace_val)
    gs_colorbar_zone = gs_master[1].subgridspec(2, 1, hspace=hspace_val)

    res_ticks = [t for t in range(0, int(global_vmax) + 30, 30) if t <= global_vmax]

    map_im_handle = None
    scatter_im_handle = None

    for col, name in enumerate(MODEL_NAMES):
        # -----------------------------------------------------------------
        # Row 0: Jason-3 Kernel Density Scatter Plot
        # -----------------------------------------------------------------
        ax_scat = fig.add_subplot(gs_left[0, col])

        x = scatter_x_pool[name]
        y = scatter_y_pool[name]

        xy = np.vstack([x, y])
        try:
            z = gaussian_kde(xy)(xy)
        except Exception:
            z = np.ones_like(x)

        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        # 渲染引擎换回统一的 viridis
        sc = ax_scat.scatter(x, y, c=z, s=2.0, cmap='viridis', alpha=0.8, edgecolor='none')
        if col == 2: scatter_im_handle = sc

        ax_scat.plot([global_vmin, global_vmax], [global_vmin, global_vmax], 'r--', linewidth=1.0, label='1:1 Line')

        ax_scat.set_xlim(global_vmin, global_vmax)
        ax_scat.set_ylim(global_vmin, global_vmax)
        ax_scat.grid(True, linestyle='--', alpha=0.3)
        ax_scat.set_aspect('equal', adjustable='box')

        rmse = np.sqrt(mean_squared_error(x, y))
        bias = np.mean(y - x)
        r2 = r2_score(x, y)
        metrics_text = f"RMSE: {rmse:.2f}\nBias: {bias:.2f}\nR²: {r2:.2f}"

        ax_scat.text(0.05, 0.95, metrics_text, transform=ax_scat.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

        if col == 0:
            ax_scat.set_ylabel("Model TEC (TECU)", fontsize=10, fontweight='bold')
        ax_scat.set_xlabel("Jason-3 TEC (TECU)", fontsize=10, fontweight='bold')
        ax_scat.tick_params(axis='both', labelsize=10)
        ax_scat.set_title(ABC_LABELS[col], loc='left', fontsize=10, fontweight='bold', pad=8)

        # -----------------------------------------------------------------
        # Row 1: Cartopy Ocean Map with Jason Squares Overlay
        # -----------------------------------------------------------------
        ax_map = fig.add_subplot(gs_left[1, col], projection=ccrs.PlateCarree())
        add_map_outline(ax_map)

        # 1. 铺设模型预测构成的连续底层场
        masked_map = np.where(mask_gold, map_data_pool[name], np.nan)
        im = ax_map.pcolormesh(lon_grid, lat_grid, masked_map, transform=ccrs.PlateCarree(),
                               cmap='viridis', vmin=global_vmin, vmax=global_vmax, shading='nearest', zorder=1)

        # 2. 悬浮点缀：将 Jason-3 的采样位置利用细白线外框的小方块(marker='s')叠加其上
        # 色素承载采样点真实 TEC 值，形成直观对比
        ax_map.scatter(jason_lons, jason_lats, c=jason_true_tec,
                       cmap='viridis', vmin=global_vmin, vmax=global_vmax,
                       s=4, marker='s', edgecolors='white', linewidths=0.3,
                       transform=ccrs.PlateCarree(), zorder=1)

        if col == 2: map_im_handle = im

        ax_map.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())
        ax_map.set_aspect('equal', adjustable='box')

        format_gridlines(ax_map, show_lon=True, show_lat=(col == 0))
        ax_map.set_title(ABC_LABELS[col + 3], loc='left', fontsize=10, fontweight='bold', pad=8)

    # =================================================================
    # 集约纵向色标处理
    # =================================================================
    # 1. Density Colorbar (Row 0)
    ax_cb_scat = fig.add_subplot(gs_colorbar_zone[0, 0])
    ax_cb_scat.axis('off')
    cax_scat = inset_axes(ax_cb_scat, width="15%", height="85%", loc='center left', borderpad=0)
    cb_scat = fig.colorbar(scatter_im_handle, cax=cax_scat, orientation='vertical')
    cb_scat.ax.tick_params(labelsize=8)

    cb_scat.locator = mticker.MaxNLocator(nbins=4)
    cb_scat.update_ticks()

    # 科学排版防止密集刻度导致右侧切边
    cb_scat.formatter = mticker.ScalarFormatter(useMathText=True)
    cb_scat.formatter.set_powerlimits((-2, 2))
    cb_scat.update_ticks()

    cb_scat.set_label('Density', rotation=90, labelpad=12, fontsize=10, fontweight='bold')

    # 2. TEC Colorbar (Row 1)
    ax_cb_map = fig.add_subplot(gs_colorbar_zone[1, 0])
    ax_cb_map.axis('off')
    cax_map = inset_axes(ax_cb_map, width="15%", height="90%", loc='center left', borderpad=0)
    cb_map = fig.colorbar(map_im_handle, cax=cax_map, orientation='vertical')
    cb_map.ax.tick_params(labelsize=10)
    cb_map.set_ticks(res_ticks)
    cb_map.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    cb_map.set_label('TEC (TECU)', rotation=90, labelpad=12, fontsize=10, fontweight='bold')

    plt.subplots_adjust(left=0.07, right=0.92, top=0.92, bottom=0.08)

    save_path = os.path.join(OUTPUT_DIR, "jason3_validation_scatter_map.png")
    plt.savefig(save_path, dpi=600, facecolor='white')
    plt.close()
    print(f"\n✅ 悬浮方块嵌套验证图已生成！请查看: {save_path}")


if __name__ == "__main__":
    main()