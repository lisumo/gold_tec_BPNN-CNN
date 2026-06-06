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
    raise ImportError("本组图排版脚本必须依赖 Cartopy 库，请检查运行环境。")

# 添加项目根目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.models.fusion import TECFusionCNNModel

import matplotlib.font_manager as fm

# 动态加载 Times New Roman 字体文件
fm.fontManager.addfont('/home/ymli/Fonts/times.ttf')

# 全局强行覆盖为标准的 10 号 Times New Roman 字体
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# =================================================================
# ⚙️ 3x6 矩阵组图配置区
# =================================================================
TARGET_TIME_STRS = ["20230218_221000", "20230218_231000", "20230219_001000"]
ROW_LABELS = ["2023-02-18\n22:10 UT", "2023-02-18\n23:10 UT", "2023-02-19\n00:10 UT"]

COL_LABELS = [
    "BPNN\nResult", "BPNN\nResidual",
    "CNN\nResult", "CNN\nResidual",
    "Fusion\nResult", "Fusion\nResidual"
]

DATASET_PATH_OVERRIDE = "/mnt/public/GOLD/2branches_BPNN+CNN/dataset/tec_2d_enhanced_dataset_test.pth"
MODEL_PATH = "/mnt/public/GOLD/2branches_BPNN+CNN/logs/exp1_fusion_full/best_model.pth"

OUTPUT_DIR = os.path.join(project_root, 'logs', 'paper_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_map_outline(ax):
    """绘制高分辨率海岸线（不画国界线）"""
    coast = NaturalEarthFeature('physical', 'coastline', '50m',
                                edgecolor='black', facecolor='none', linewidth=0.4, zorder=3)
    ax.add_feature(coast)


def format_gridlines(ax, show_lon=True, show_lat=True):
    """精细化控制矩阵内各子图的经纬度刻度消融策略"""
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
    print(f"🚀 [Phase 1/2] 全量模型推理与全平面动态最值收集...")

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
            raise ValueError(f"测试集中未匹配到目标时刻: {target_ts}")

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

        res_pvs = [map_bpnn[mask_gold], map_cnn[mask_gold], map_fusion[mask_gold]]
        diff_pvs = [(map_bpnn - map_true)[mask_inter], (map_cnn - map_true)[mask_inter],
                    (map_fusion - map_true)[mask_inter]]

        for arr in res_pvs: gathered_res_values.extend(arr[~np.isnan(arr)])
        for arr in diff_pvs: gathered_diff_values.extend(arr[~np.isnan(arr)])

        processed_matrix_pool[t_str] = {
            'bpnn': map_bpnn, 'cnn': map_cnn, 'fusion': map_fusion,
            'true': map_true, 'mask_gold': mask_gold, 'mask_inter': mask_inter
        }

    global_res_vmin = 0.0
    global_res_vmax = float(np.max(gathered_res_values))

    global_diff_raw_min = float(np.min(gathered_diff_values))
    global_diff_raw_max = float(np.max(gathered_diff_values))
    global_diff_limit = max(abs(global_diff_raw_min), abs(global_diff_raw_max))
    global_diff_vmin = -global_diff_limit
    global_diff_vmax = global_diff_limit

    # =================================================================
    # 🚀 [Phase 2/2] 高标准排版画布布局精准控制
    # =================================================================
    print(f"\n🎨 [Phase 2/2] 启动右舷格网吸附重构机制...")

    dummy_key = TARGET_TIME_STRS[0]
    map_shape = processed_matrix_pool[dummy_key]['true'].shape
    lon_arr = np.linspace(Config.MASK_LON_MIN, Config.MASK_LON_MAX, map_shape[1])
    lat_arr = np.linspace(Config.MASK_LAT_MIN, Config.MASK_LAT_MAX, map_shape[0])
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    a4_width_inch = 6.27
    fig_height = 3.3
    fig = plt.figure(figsize=(a4_width_inch, fig_height))

    from matplotlib.gridspec import GridSpec
    # 【重大重构 1】母网格降维为 1x2 拓扑，完全解放左、右两边的耦合约束
    # wspace=0.12 控制时刻容器与大图的舒服空白，永远不会被打乱
    gs_master = GridSpec(1, 2, figure=fig, width_ratios=[0.35, 5.92], wspace=0.12)

    wspace_val = 0.04
    hspace_val = 0.13

    gs_left = gs_master[0].subgridspec(3, 1, hspace=hspace_val)

    # 【重大重构 2】在母网格右路建立独立的“联合视口舱”
    # 通过将这里的 wspace 精准锁死在极低的 0.01 到 0.02，实现色条向左“瞬间吸附吸附”！
    gs_right_container = gs_master[1].subgridspec(1, 2, width_ratios=[5.32, 0.60], wspace=0.01)

    # 从联合舱中并行解包出主图和色条网格
    gs_right = gs_right_container[0].subgridspec(3, 6, wspace=wspace_val, hspace=hspace_val)
    gs_colorbar_zone = gs_right_container[1].subgridspec(3, 2, wspace=0.70, hspace=hspace_val)

    res_ticks = [t for t in range(0, int(global_res_vmax) + 30, 30) if t <= global_res_vmax]
    diff_ticks = [t for t in range(-120, 121, 30) if global_diff_vmin <= t <= global_diff_vmax]

    for row, t_str in enumerate(TARGET_TIME_STRS):
        m_data = processed_matrix_pool[t_str]

        matrix_channels = [
            (np.where(m_data['mask_gold'], m_data['bpnn'], np.nan), 'viridis', global_res_vmin, global_res_vmax),
            (np.where(m_data['mask_inter'], m_data['bpnn'] - m_data['true'], np.nan), 'RdBu_r', global_diff_vmin,
             global_diff_vmax),
            (np.where(m_data['mask_gold'], m_data['cnn'], np.nan), 'viridis', global_res_vmin, global_res_vmax),
            (np.where(m_data['mask_inter'], m_data['cnn'] - m_data['true'], np.nan), 'RdBu_r', global_diff_vmin,
             global_diff_vmax),
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

            if col == 4: row_res_im = im
            if col == 5: row_diff_im = im

            ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())
            ax.set_aspect('equal', adjustable='box')

            show_latitude = (col == 0)
            show_longitude = (row == 2)
            format_gridlines(ax, show_lon=show_longitude, show_lat=show_latitude)

            if row == 0:
                ax.set_title(COL_LABELS[col], fontsize=10, pad=10, fontweight='bold')

        # =================================================================
        # 🎨 内置高精度色条控制
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

    # 【完美收边】由于右侧大矩阵整体无损向左侧贴紧，我们把画布的最右边调整到 0.96，让画面张力完全拉满
    plt.subplots_adjust(left=0.01, right=0.96, top=0.86, bottom=0.10)

    save_path = os.path.join(OUTPUT_DIR, "quiet_period_matrix_analysis.png")
    plt.savefig(save_path, dpi=600, facecolor='white')
    plt.close()
    print(f"\n✅ 联合舱隔绝优化版编译成功！大图与色条的悬空鸿沟已被彻底压平。文件输出至:\n👉 {save_path}")


if __name__ == "__main__":
    main()