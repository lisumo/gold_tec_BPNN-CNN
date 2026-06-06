import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 强制设置无GUI后端
import matplotlib

matplotlib.use('Agg')

try:
    import cartopy.crs as ccrs
    from cartopy.feature import NaturalEarthFeature

    HAS_CARTOPY = True
except ImportError:
    raise ImportError("Cartopy required.")

# 添加项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.models.fusion import TECFusionCNNModel

import matplotlib.font_manager as fm

fm.fontManager.addfont('/home/ymli/Fonts/times.ttf')

# 全局 rcParams 控制
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# =================================================================
# ⚙️ 陆地遮挡组图配置区
# =================================================================
TARGET_TIME_STR = "20230408_001000"
DATASET_PATH_OVERRIDE = "/mnt/public/GOLD/2branches_BPNN+CNN/dataset/tec_2d_enhanced_dataset_test.pth"

MODEL_PATHS = [
    "/mnt/public/GOLD/2branches_BPNN+CNN/logs/exp5_mask_R1/best_model.pth",
    "/mnt/public/GOLD/2branches_BPNN+CNN/logs/exp5_mask_R2/best_model.pth",
    "/mnt/public/GOLD/2branches_BPNN+CNN/logs/exp5_mask_R3/best_model.pth",
    "/mnt/public/GOLD/2branches_BPNN+CNN/logs/exp5_mask_R4/best_model.pth",
    "/mnt/public/GOLD/2branches_BPNN+CNN/logs/exp5_mask_R5/best_model.pth",
    "/mnt/public/GOLD/2branches_BPNN+CNN/logs/exp5_mask_R6/best_model.pth"
]

OUTPUT_DIR = os.path.join(project_root, 'logs', 'paper_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def extract_model_prediction(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print(f"Warning: model file not found {ckpt_path}")
        return None, None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    scaler_y = checkpoint['scaler_y']
    scaler_X_spatial = checkpoint['scaler_X_spatial']
    scaler_X_time = checkpoint['scaler_X_time']

    raw_dataset_path = DATASET_PATH_OVERRIDE if DATASET_PATH_OVERRIDE is not None else Config.DATASET_PATH
    if not os.path.isabs(raw_dataset_path):
        raw_dataset_path = os.path.join(project_root, raw_dataset_path)

    if os.path.exists(raw_dataset_path) and os.path.isfile(raw_dataset_path):
        test_data_path = raw_dataset_path
    else:
        base, ext = os.path.splitext(raw_dataset_path)
        test_data_path = f"{base}_test{ext}"

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Dataset file not found! {test_data_path}")

    samples = torch.load(test_data_path, weights_only=False)
    model = TECFusionCNNModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    try:
        if "_" in str(TARGET_TIME_STR):
            target_timestamp = pd.to_datetime(TARGET_TIME_STR, format="%Y%m%d_%H%M%S")
        else:
            target_timestamp = pd.to_datetime(TARGET_TIME_STR)
    except Exception as e:
        raise ValueError(f"Failed to parse time: {e}")

    target_sample = None
    for s in samples:
        time_label = s[6] if len(s) > 6 else ""
        try:
            if "_" in str(time_label):
                current_timestamp = pd.to_datetime(time_label, format="%Y%m%d_%H%M%S")
            else:
                current_timestamp = pd.to_datetime(time_label)

            if current_timestamp == target_timestamp:
                target_sample = s
                break
        except (ValueError, TypeError):
            continue

    if target_sample is None:
        raise ValueError(f"Target time not found in dataset.")

    with torch.no_grad():
        sp = target_sample[0].numpy()
        C, H, W = sp.shape
        sp_norm = scaler_X_spatial.transform(sp.reshape(1, -1)).reshape(C, H, W)
        tm = target_sample[1].numpy().reshape(1, -1)
        tm_norm = scaler_X_time.transform(tm).reshape(-1)

        b_sp = torch.FloatTensor(sp_norm).unsqueeze(0).to(device)
        b_tm = torch.FloatTensor(tm_norm).unsqueeze(0).to(device)

        final_out, _, _ = model(b_sp, b_tm)
        pred_arr = final_out.cpu().numpy().reshape(1, -1)
        map_pred = scaler_y.inverse_transform(pred_arr).reshape(config['input_shape'])

        mask_gold = target_sample[5].numpy().reshape(config['input_shape']).astype(bool)
        map_pred_masked = np.where(mask_gold, map_pred, np.nan)

        map_true = target_sample[2].numpy().reshape(config['input_shape'])
        mask_mit = target_sample[3].numpy().reshape(config['input_shape']).astype(bool)
        map_true_masked = np.where(mask_mit, map_true, np.nan)

    return map_pred_masked, map_true_masked


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting re-plotting flow. Target time: {TARGET_TIME_STR}")

    pred_maps = []
    map_true_ref = None

    for path in MODEL_PATHS:
        print(f"Processing model: {os.path.basename(os.path.dirname(path))}")
        pred_map, true_map = extract_model_prediction(path, device)
        if pred_map is not None:
            pred_maps.append(pred_map)
            if map_true_ref is None:
                map_true_ref = true_map

    lon_arr = np.linspace(Config.MASK_LON_MIN, Config.MASK_LON_MAX, map_true_ref.shape[1])
    lat_arr = np.linspace(Config.MASK_LAT_MIN, Config.MASK_LAT_MAX, map_true_ref.shape[0])
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    a4_width_inch = 6.27
    fig_height = 3.6

    fig = plt.figure(figsize=(a4_width_inch, fig_height))

    from matplotlib.gridspec import GridSpec
    wspace_val = 0.05
    hspace_val = 0.16  # ⚡ 行间距呼吸感释放（增加一倍）
    cb_width_ratio = 0.35  # 色条专属计算基底

    # 📐 【核心公式：无损数学解耦】
    # 左系统：图占 1.0 + 色条占 cb_width_ratio
    left_width = 1.0 + cb_width_ratio

    # 右系统：3个图占 3.0 + 2个列间隙 + 色条占 cb_width_ratio
    right_width = 3.0 + 2.0 * wspace_val + cb_width_ratio

    gs_master = GridSpec(1, 2, figure=fig, width_ratios=[left_width, right_width], wspace=0.15)

    # 左侧舱建立（子图列+色条列）
    gs_left_container = gs_master[0].subgridspec(1, 2, width_ratios=[1.0, cb_width_ratio], wspace=0.04)
    pad_ratio = (1.0 + hspace_val) / 2.0
    gs_left_map = gs_left_container[0].subgridspec(3, 1, height_ratios=[pad_ratio, 1.0, pad_ratio], hspace=0)
    gs_left_cb = gs_left_container[1].subgridspec(3, 1, height_ratios=[pad_ratio, 1.0, pad_ratio], hspace=0)

    # 右侧舱建立（矩阵列+色条列）
    gs_right_container = gs_master[1].subgridspec(1, 2, width_ratios=[3.0 + 2.0 * wspace_val, cb_width_ratio],
                                                  wspace=0.04)
    gs_right = gs_right_container[0].subgridspec(2, 3, wspace=wspace_val, hspace=hspace_val)
    gs_colorbar_zone = gs_right_container[1].subgridspec(2, 1, hspace=hspace_val)

    vmin, vmax = 0, 80
    abc_idx = 0
    abc_list = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    # =================================================================
    # 🖼️ 左侧：MIT Ground Truth Map
    # =================================================================
    ax_truth = fig.add_subplot(gs_left_map[1, 0], projection=ccrs.PlateCarree())
    add_map_outline(ax_truth)

    im_truth = ax_truth.pcolormesh(lon_grid, lat_grid, map_true_ref, transform=ccrs.PlateCarree(),
                                   cmap='viridis', vmin=vmin, vmax=vmax, shading='nearest', zorder=1)

    ax_truth.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())
    ax_truth.set_aspect('equal', adjustable='box')

    format_gridlines(ax_truth, show_lon=True, show_lat=True)
    ax_truth.set_title(abc_list[abc_idx], loc='left', fontweight='bold', pad=6)
    abc_idx += 1

    # 🎨 【恢复左侧独立色条】：物理厚度压缩至 25%
    ax_cb_truth = fig.add_subplot(gs_left_cb[1, 0])
    ax_cb_truth.axis('off')
    cax_truth = inset_axes(ax_cb_truth, width="15%", height="88%", loc='center left', borderpad=0.3)
    cb_truth = fig.colorbar(im_truth, cax=cax_truth, orientation='vertical')
    cb_truth.ax.tick_params(labelsize=8)
    cb_truth.set_ticks(np.arange(0, vmax + 10, 20))
    cb_truth.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    cb_truth.set_label('TEC (TECU)', rotation=90, labelpad=2, fontsize=8, fontweight='bold')

    # =================================================================
    # 🖼️ 右侧：2x3 Array (R1 ~ R6)
    # =================================================================
    model_idx = 0
    row_last_ims = [None, None]

    for row in range(2):
        for col in range(3):
            if model_idx >= len(pred_maps): break

            ax = fig.add_subplot(gs_right[row, col], projection=ccrs.PlateCarree())
            add_map_outline(ax)

            im = ax.pcolormesh(lon_grid, lat_grid, pred_maps[model_idx], transform=ccrs.PlateCarree(),
                               cmap='viridis', vmin=vmin, vmax=vmax, shading='nearest', zorder=1)

            if col == 2:
                row_last_ims[row] = im

            ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())
            ax.set_aspect('equal', adjustable='box')

            show_latitude = (col == 0)
            show_longitude = (row == 1)
            format_gridlines(ax, show_lon=show_longitude, show_lat=show_latitude)

            ax.set_title(abc_list[abc_idx], loc='left', fontweight='bold', pad=6)
            abc_idx += 1
            model_idx += 1

    # =================================================================
    # 🎨 右侧：每行独立的右置集约色条 (极细化 25%)
    # =================================================================
    for row in range(2):
        ax_cb = fig.add_subplot(gs_colorbar_zone[row, 0])
        ax_cb.axis('off')

        cax = inset_axes(ax_cb, width="15%", height="88%", loc='center left', borderpad=0)
        cb = fig.colorbar(row_last_ims[row], cax=cax, orientation='vertical')

        cb.ax.tick_params(labelsize=8)
        cb.set_ticks(np.arange(0, vmax + 10, 20))
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        cb.set_label('TEC (TECU)', rotation=90, labelpad=4, fontsize=8, fontweight='bold')

    # ⚡【整体重心右移】：left 调大防止左边裁切，right 收缩腾出右侧空白
    plt.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.10)

    save_path = os.path.join(OUTPUT_DIR, f"combined_spatial_analysis_{TARGET_TIME_STR}.png")
    plt.savefig(save_path, dpi=600, facecolor='white')
    plt.close()
    print(f"\n✅ 陆地遮挡实验完美恢复版！左侧色条补齐，图大一致，边界偏移已修复。输出至:\n👉 {save_path}")


if __name__ == "__main__":
    main()