import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Force non-GUI backend
import matplotlib

matplotlib.use('Agg')

try:
    import cartopy.crs as ccrs
    from cartopy.feature import NaturalEarthFeature

    HAS_CARTOPY = True
except ImportError:
    raise ImportError("Cartopy is required for this script.")

# Add project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.models.fusion import TECFusionCNNModel

import matplotlib.font_manager as fm

# 动态加载 Times New Roman 字体文件
fm.fontManager.addfont('/home/ymli/Fonts/times.ttf')

# Global rcParams update
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# =================================================================
# Configuration Zone for 3x5 Disturbance Matrix Plot
# =================================================================
TARGET_TIME_STRS = ["20231212_232200", "20231212_235200", "20231213_002200"]
ROW_LABELS = ["2023-12-12\n23:22 UT", "2023-12-12\n23:52 UT", "2023-12-13\n00:22 UT"]

COL_LABELS = [
    "MIT\nTEC", "BPNN\nBranch",
    "CNN\nBranch", "Fusion\nResult",
    "Fusion\nResidual"
]

DATASET_PATH_OVERRIDE = "/mnt/public/GOLD/2branches_BPNN+CNN/dataset/tec_2d_enhanced_dataset_test.pth"
MODEL_PATH = "/mnt/public/GOLD/2branches_BPNN+CNN/logs/exp1_fusion_full/best_model.pth"

OUTPUT_DIR = os.path.join(project_root, 'logs', 'paper_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_map_outline(ax):
    # 仅绘制高分辨率海岸线，彻底清空国界轮廓线
    coast = NaturalEarthFeature('physical', 'coastline', '50m',
                                edgecolor='black', facecolor='none', linewidth=0.4, zorder=3)
    ax.add_feature(coast)


def format_gridlines(ax, show_lon=True, show_lat=True):
    # 矩阵经纬度标签全表面消融策略控制
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
    print("🚀 [Phase 1/2] Processing inference and gathering dynamic limits...")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = checkpoint['config']
    scaler_y = checkpoint['scaler_y']
    scaler_X_spatial = checkpoint['scaler_X_spatial']
    scaler_X_time = checkpoint['scaler_X_time']

    model = TECFusionCNNModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    samples = torch.load(DATASET_PATH_OVERRIDE, weights_only=False)
    processed_matrix_pool = {}

    gathered_res_values = []
    gathered_diff_values = []

    for t_str in TARGET_TIME_STRS:
        target_ts = pd.to_datetime(t_str, format="%Y%m%d_%H%M%S")
        target_sample = None
        for s in samples:
            time_label = s[6] if len(s) > 6 else ""
            try:
                curr_ts = pd.to_datetime(time_label, format="%Y%m%d_%H%M%S") if "_" in str(
                    time_label) else pd.to_datetime(time_label)
                if curr_ts == target_ts:
                    target_sample = s
                    break
            except (ValueError, TypeError):
                continue

        if target_sample is None:
            raise ValueError(f"Target timestamp not found in test set: {target_ts}")

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

            map_fusion = inverse_to_map(final_out)
            map_cnn = inverse_to_map(cnn_out)
            map_bpnn = inverse_to_map(bpnn_out)

            map_true = target_sample[2].numpy().reshape(config['input_shape'])
            mask_mit = target_sample[3].numpy().reshape(config['input_shape']).astype(bool)
            mask_gold = target_sample[5].numpy().reshape(config['input_shape']).astype(bool)
            mask_inter = mask_mit & mask_gold

        # Column 0: MIT, Column 1-3: Predictions, Column 4: Residual
        res_pvs = [map_true[mask_mit], map_bpnn[mask_gold], map_cnn[mask_gold], map_fusion[mask_gold]]
        diff_pvs = [(map_fusion - map_true)[mask_inter]]

        for arr in res_pvs: gathered_res_values.extend(arr[~np.isnan(arr)])
        for arr in diff_pvs: gathered_diff_values.extend(arr[~np.isnan(arr)])

        processed_matrix_pool[t_str] = {
            'bpnn': map_bpnn, 'cnn': map_cnn, 'fusion': map_fusion,
            'true': map_true, 'mask_mit': mask_mit, 'mask_gold': mask_gold, 'mask_inter': mask_inter
        }

    # 固定结果图最小值为 0.0
    global_res_vmin = 0.0
    global_res_vmax = float(np.max(gathered_res_values))

    # 零点对称，保护发散色标白色中线
    global_diff_raw_min = float(np.min(gathered_diff_values))
    global_diff_raw_max = float(np.max(gathered_diff_values))
    global_diff_limit = max(abs(global_diff_raw_min), abs(global_diff_raw_max))
    global_diff_vmin = -global_diff_limit
    global_diff_vmax = global_diff_limit

    # =================================================================
    # [Phase 2/2] Precise Geometric Layout Construction (3x5 Matrix)
    # =================================================================
    print("\n🎨 [Phase 2/2] 释放纵向瓶颈并启动极限地图面积扩容渲染...")

    dummy_key = TARGET_TIME_STRS[0]
    map_shape = processed_matrix_pool[dummy_key]['true'].shape
    lon_arr = np.linspace(Config.MASK_LON_MIN, Config.MASK_LON_MAX, map_shape[1])
    lat_arr = np.linspace(Config.MASK_LAT_MIN, Config.MASK_LAT_MAX, map_shape[0])
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    a4_width_inch = 6.27
    # 【⚡关键修改1】将画布高度大幅调高至 3.8 英寸，彻底解除正方形生长的纵向短板
    fig_height = 3.8
    fig = plt.figure(figsize=(a4_width_inch, fig_height))

    from matplotlib.gridspec import GridSpec
    # 1x2 主格网隔离
    gs_master = GridSpec(1, 2, figure=fig, width_ratios=[0.35, 5.92], wspace=0.12)

    # 【⚡关键修改2】将间距全面“脱水压缩”，把空隙像素全额返还给子图本身
    wspace_val = 0.02  # 横向列间距从 0.04 收紧到 0.02
    hspace_val = 0.08  # 纵向行间距从 0.13 压缩到 0.08，使得图之间极其致密

    gs_left = gs_master[0].subgridspec(3, 1, hspace=hspace_val)
    gs_right_container = gs_master[1].subgridspec(1, 2, width_ratios=[5.32, 0.60], wspace=0.01)

    gs_right = gs_right_container[0].subgridspec(3, 5, wspace=wspace_val, hspace=hspace_val)
    gs_colorbar_zone = gs_right_container[1].subgridspec(3, 2, wspace=0.70, hspace=hspace_val)

    # 30 纯整数固定步长
    res_ticks = [t for t in range(0, int(global_res_vmax)-10, 30) if t <= global_res_vmax]
    diff_ticks = [t for t in range(-120, 121, 30) if global_diff_vmin <= t <= global_diff_vmax]

    for row, t_str in enumerate(TARGET_TIME_STRS):
        m_data = processed_matrix_pool[t_str]

        matrix_channels = [
            (np.where(m_data['mask_mit'], m_data['true'], np.nan), 'viridis', global_res_vmin, global_res_vmax),
            (np.where(m_data['mask_gold'], m_data['bpnn'], np.nan), 'viridis', global_res_vmin, global_res_vmax),
            (np.where(m_data['mask_gold'], m_data['cnn'], np.nan), 'viridis', global_res_vmin, global_res_vmax),
            (np.where(m_data['mask_gold'], m_data['fusion'], np.nan), 'viridis', global_res_vmin, global_res_vmax),
            (np.where(m_data['mask_inter'], m_data['fusion'] - m_data['true'], np.nan), 'RdBu_r', global_diff_vmin,
             global_diff_vmax)
        ]

        # 竖排时刻渲染
        ax_label_container = fig.add_subplot(gs_left[row, 0])
        ax_label_container.axis('off')
        ax_label_container.text(0.5, 0.5, ROW_LABELS[row], transform=ax_label_container.transAxes,
                                rotation=90, va='center', ha='center', fontweight='bold', fontsize=10)

        row_res_im = None
        row_diff_im = None

        for col, (data, cmap, vmin, vmax) in enumerate(matrix_channels):
            ax = fig.add_subplot(gs_right[row, col], projection=ccrs.PlateCarree())
            add_map_outline(ax)

            im = ax.pcolormesh(lon_grid, lat_grid, data, transform=ccrs.PlateCarree(),
                               cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest', zorder=1)

            if col == 3: row_res_im = im
            if col == 4: row_diff_im = im

            ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())
            ax.set_aspect('equal', adjustable='box')

            show_latitude = (col == 0)
            show_longitude = (row == 2)
            format_gridlines(ax, show_lon=show_longitude, show_lat=show_latitude)

            if row == 0:
                ax.set_title(COL_LABELS[col], fontsize=10, pad=10, fontweight='bold')

        # =================================================================
        # Mini-Inset-Axes 嵌入式窄色标控制（纯整数 + 8pt 字号）
        # =================================================================
        # 1. 结果图色条 (TEC)
        ax_cb_res = fig.add_subplot(gs_colorbar_zone[row, 0])
        ax_cb_res.axis('off')
        cax_res = inset_axes(ax_cb_res, width="30%", height="95%", loc='center left', borderpad=0)
        cb_res = fig.colorbar(row_res_im, cax=cax_res, orientation='vertical')
        cb_res.ax.tick_params(labelsize=8)
        cb_res.set_ticks(res_ticks)
        cb_res.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        if row == 0:
            cax_res.set_title('TEC\n(TECU)', fontsize=7, pad=6)

        # 2. 残差图色条 (ΔTEC)
        ax_cb_diff = fig.add_subplot(gs_colorbar_zone[row, 1])
        ax_cb_diff.axis('off')
        cax_diff = inset_axes(ax_cb_diff, width="30%", height="95%", loc='center left', borderpad=0)
        cb_diff = fig.colorbar(row_diff_im, cax=cax_diff, orientation='vertical')
        cb_diff.ax.tick_params(labelsize=8)
        cb_diff.set_ticks(diff_ticks)
        cb_diff.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        if row == 0:
            cax_diff.set_title('ΔTEC\n(TECU)', fontsize=7, pad=6)

    # 外边缘极致切边
    plt.subplots_adjust(left=0.01, right=0.96, top=0.86, bottom=0.10)

    save_path = os.path.join(OUTPUT_DIR, "disturbance_period_3x5_matrix.png")
    plt.savefig(save_path, dpi=600, facecolor='white')
    plt.close()
    print(f"\\n✅ 3x5 强扰动大面积版矩阵图重构大成功！图片输出至:\\n👉 {save_path}")


if __name__ == "__main__":
    main()