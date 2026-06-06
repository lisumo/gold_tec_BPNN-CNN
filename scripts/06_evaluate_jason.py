import matplotlib

matplotlib.use('Agg')  # 服务器无头模式

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.utils.common import setup_gpu
from src.models.fusion import TECFusionCNNModel
from src.utils.options import parse_args
from src.data_process.jason_parser import find_jason_files_for_timestamp, read_and_filter_jason_data

# 设置绘图风格 (与 05_evaluate_server.py 保持一致)
try:
    import cartopy.crs as ccrs
    from cartopy.feature import NaturalEarthFeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️ 警告: 未检测到 Cartopy，地图底图将无法绘制！")

# 设置全局字体风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# =================================================================
# 🛠️ 模型配置区域 (请在此处修改您的实验名称)
# =================================================================
MODELS_CONFIG = {
    'fusion': 'exp1_fusion_full',  # 融合模型实验名
    'bpnn': 'exp3_bpnn_only',  # BPNN模型实验名 (请修改)
    'cnn': 'exp2_cnn_only'  # CNN模型实验名 (请修改)
}


# =================================================================

def get_physical_grid(input_shape):
    """
    根据 Config 和 Input Shape 重建物理格网
    """
    # 获取经纬度范围
    lat_min, lat_max = Config.MASK_LAT_MIN, Config.MASK_LAT_MAX
    lon_min, lon_max = Config.MASK_LON_MIN, Config.MASK_LON_MAX

    H, W = input_shape

    # 使用 linspace 生成格网中心点
    # 注意：这里生成的 lats/lons 是网格中心坐标
    lats = np.linspace(lat_min, lat_max, H)
    lons = np.linspace(lon_min, lon_max, W)

    return lats, lons


# =================================================================
# 🎨 绘图功能函数 (核心修改：使用 pcolormesh + NaN Mask)
# =================================================================
def plot_overlay_map(model_grid, valid_df, gold_mask, lats, lons, time_str, save_dir, model_name):
    """
    绘制卫星轨迹叠加图
    """
    if not HAS_CARTOPY: return

    # 【修改】图尺寸与散点图一致
    fig = plt.figure(figsize=(2.36, 2.36))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 【修改】地图要素：无填充，仅轮廓线
    ax.add_feature(NaturalEarthFeature('physical', 'land', '50m',
                                       edgecolor='black', facecolor='none',
                                       linewidth=0.3))
    ax.coastlines(linewidth=0.5, color='black')

    # 【修改】网格线字体5pt
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5,
                      linewidth=0.4, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 5}
    gl.ylabel_style = {'size': 5}

    # 准备网格坐标
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    vmin, vmax = 0, 60

    # 处理mask（保持原逻辑）
    if torch.is_tensor(gold_mask):
        mask_np = gold_mask.detach().cpu().numpy().squeeze()
    else:
        mask_np = gold_mask.squeeze()

    plot_grid = model_grid.copy()
    plot_grid[mask_np < 0.5] = np.nan

    # 绘制模型背景（pcolormesh）
    im = ax.pcolormesh(lon_grid, lat_grid, plot_grid,
                       transform=ccrs.PlateCarree(),
                       cmap='viridis', vmin=vmin, vmax=vmax,
                       shading='nearest')

    # 绘制Jason轨迹（方块marker）
    sc = ax.scatter(valid_df['lon'], valid_df['lat'], c=valid_df['jason_tec'],
                    transform=ccrs.PlateCarree(), cmap='viridis',
                    vmin=vmin, vmax=vmax,
                    edgecolors='white', linewidth=0.3, s=5, marker='s',
                    label='Jason-3')

    # 【修改】紧凑colorbar，字体6pt
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('TEC (TECU)', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # 【删除】取消标题
    # ax.set_title(...)  # 已删除

    # 【修改】图例左下角，字体6pt
    ax.legend(loc='lower left', framealpha=0.9, fontsize=6,
              fancybox=False, edgecolor='gray')

    # 【修改】设置范围，保持紧凑
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())

    # 【修改】调整布局，减少白边
    plt.tight_layout(pad=0.3)

    # 保存
    safe_name = time_str.replace(':', '-').replace(' ', '_')
    filename = f"map_{model_name}_{safe_name}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=600,
                bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_global_scatter(df, pred_col, save_dir, model_name):
    """绘制全局散点回归图（密度着色版本，适配三图并排布局）"""
    if df.empty: return

    from scipy.stats import gaussian_kde
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import FuncFormatter  # 【新增】导入格式化器

    # 图尺寸：宽度包含colorbar空间，高度确保坐标轴区域为正方形
    fig_width = 2.36
    fig_height = 2.6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = df['jason_tec']
    y = df[pred_col]

    # 计算指标
    rmse = np.sqrt(mean_squared_error(x, y))
    bias = np.mean(y - x)
    r2 = 1 - (np.sum((y - x) ** 2) / np.sum((x - np.mean(x)) ** 2))

    # 计算数据点密度
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    density = kde(xy)

    # 散点尺寸调小以适应小图，密度着色
    scatter = ax.scatter(x, y, c=density, s=5, cmap='viridis',
                         edgecolors='none', alpha=0.8)

    # 绘制 1:1 线（线宽调细）
    limit = max(np.nanmax(x), np.nanmax(y)) + 5
    ax.plot([0, limit], [0, limit], 'r--', linewidth=1, label='1:1 Line')

    # 确保坐标轴区域为正方形
    ax.set_aspect('equal', adjustable='box')

    # 坐标轴范围与刻度
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)

    # 刻度间隔自适应，字体5-6pt
    ax.tick_params(axis='both', labelsize=5)

    # 坐标轴标签，字体7pt
    ax.set_xlabel('Jason-3 TEC (TECU)', fontsize=6)
    ax.set_ylabel(f'{model_name} TEC (TECU)', fontsize=6)

    # 网格线调细
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

    # 使用make_axes_locatable添加colorbar，不挤压坐标轴
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('Density', fontsize=6)

    # 【修改】colorbar刻度格式化为百分数，节省空间
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 100:.2f}%'))
    cbar.ax.tick_params(labelsize=5)

    # 统计文本字体6pt，位置微调避免遮挡
    stats_text = f"RMSE: {rmse:.2f}\nBias: {bias:.2f}\nR²: {r2:.3f}"
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=6, verticalalignment='top')

    # 图例字体6pt，位置紧凑
    ax.legend(loc='lower right', fontsize=6, frameon=True,
              fancybox=False, edgecolor='gray')

    # 调整布局，减少白边
    plt.tight_layout(pad=0.3)

    filename = f"global_scatter_{model_name}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"✅ 散点图已保存: {filename} (坐标轴区域为正方形，colorbar百分数格式)")


# =================================================================
# 核心逻辑：格网化匹配 (Binning)
# =================================================================
def match_jason_to_grid_multi(jason_df, model_grids, gold_mask, lat_arr, lon_arr):
    """
    将 Jason 轨迹点映射到模型的物理格网中，并进行格网化平均。
    """
    first_grid = next(iter(model_grids.values()))
    H, W = first_grid.shape

    lat_min, lat_max = lat_arr[0], lat_arr[-1]
    lon_min, lon_max = lon_arr[0], lon_arr[-1]

    lat_step = (lat_max - lat_min) / (H - 1) if H > 1 else 1.0
    lon_step = (lon_max - lon_min) / (W - 1) if W > 1 else 1.0

    # 1. 计算索引
    jason_lats = jason_df['lat'].values
    jason_lons = jason_df['lon'].values

    idx_lat = np.round((jason_lats - lat_min) / lat_step).astype(int)
    idx_lon = np.round((jason_lons - lon_min) / lon_step).astype(int)

    # 2. 过滤出界点
    valid_indices = (idx_lat >= 0) & (idx_lat < H) & (idx_lon >= 0) & (idx_lon < W)
    if not np.any(valid_indices):
        return pd.DataFrame()

    filtered_df = jason_df.iloc[valid_indices].copy()
    filtered_df['idx_lat'] = idx_lat[valid_indices]
    filtered_df['idx_lon'] = idx_lon[valid_indices]

    # 3. 格网化聚合 (Binning)
    binned_df = filtered_df.groupby(['idx_lat', 'idx_lon'])['jason_tec'].mean().reset_index()

    # 4. 提取模型值和掩码
    rows = binned_df['idx_lat'].values
    cols = binned_df['idx_lon'].values

    if torch.is_tensor(gold_mask):
        gold_mask = gold_mask.detach().cpu().numpy().squeeze()

    binned_df['gold_validity'] = gold_mask[rows, cols]

    for name, grid in model_grids.items():
        if torch.is_tensor(grid):
            grid = grid.detach().cpu().numpy().squeeze()
        binned_df[f'pred_{name}'] = grid[rows, cols]

    # 5. 恢复中心经纬度
    binned_df['lat'] = lat_arr[rows]
    binned_df['lon'] = lon_arr[cols]

    return binned_df


def load_single_model(exp_name, device):
    """
    加载单个模型 (标准加载)
    """
    log_dir = os.path.join(project_root, 'logs', exp_name)
    model_path = os.path.join(log_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"⚠️ 警告: 模型文件不存在: {model_path}")
        return None, None

    print(f"📂 Loading model: {model_path}")
    try:
        # 恢复标准加载，不使用 map_location='cpu'
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint['config']
    if isinstance(model_config.get('input_shape'), list):
        model_config['input_shape'] = tuple(model_config['input_shape'])

    # 初始化模型结构
    model = TECFusionCNNModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def evaluate_jason_pipeline():
    args = parse_args()
    JASON_DATA_DIR = "/mnt/public/Jason3_GDR"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = setup_gpu()

    # 1. 加载所有模型
    loaded_models = {}
    master_checkpoint = None

    for name, exp_name in MODELS_CONFIG.items():
        model, ckpt = load_single_model(exp_name, device)
        if model is not None:
            loaded_models[name] = model
            if master_checkpoint is None:
                master_checkpoint = ckpt

    if not loaded_models:
        print("❌ 未加载到任何模型")
        return

    scaler_y = master_checkpoint['scaler_y']
    scaler_X_spatial = master_checkpoint.get('scaler_X_spatial')
    scaler_X_time = master_checkpoint.get('scaler_X_time')
    input_shape = master_checkpoint['config']['input_shape']

    # 获取物理格网坐标
    lats, lons = get_physical_grid(input_shape)

    # 2. 全量数据加载
    print("🔄 正在加载全量数据集 (Train + Val + Test)...")
    full_samples = []

    if Config.USE_PERIODIC_SPLIT:
        base_path = Config.DATASET_PATH
        base_name, ext = os.path.splitext(base_path)
        partitions = ['train', 'val', 'test']
        for p in partitions:
            p_path = f"{base_name}_{p}{ext}"
            if os.path.exists(p_path):
                print(f"   -> Loading {p_path} ...")
                try:
                    p_data = torch.load(p_path, weights_only=False)
                except Exception:
                    p_data = torch.load(p_path)
                full_samples.extend(p_data)
    else:
        if os.path.exists(Config.DATASET_PATH):
            print(f"   -> Loading {Config.DATASET_PATH} ...")
            try:
                full_samples = torch.load(Config.DATASET_PATH, weights_only=False)
            except Exception:
                full_samples = torch.load(Config.DATASET_PATH)
        else:
            print(f"❌ Error: Dataset not found at {Config.DATASET_PATH}")
            return

    print(f"✅ 数据加载完成，共 {len(full_samples)} 个样本")

    eval_dir = os.path.join(project_root, 'logs', 'jason_comparison_results')
    plots_dir = os.path.join(eval_dir, 'plots')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    results_list = []
    target_lon_range = (-80, -20)
    target_lat_range = (0, 40)

    print(f"🚀 开始多模型对比验证 (Style: Blocky Grid & Valid Mask)...")

    for i, sample in enumerate(full_samples):
        sp_raw = sample[0].numpy()
        tm_raw = sample[1].numpy()
        gold_valid_mask = sample[5].numpy().squeeze()
        time_str = sample[6]

        matched_files = find_jason_files_for_timestamp(JASON_DATA_DIR, time_str, tolerance_minutes=30)
        if not matched_files: continue

        jason_df = read_and_filter_jason_data(
            matched_files, time_str,
            lon_range=target_lon_range,
            lat_range=target_lat_range,
            time_tol_minutes=Config.TIME_TOLERANCE_MINUTES
        )
        if len(jason_df) == 0: continue

        # 模型推理
        C, H, W = sp_raw.shape
        sp_norm = scaler_X_spatial.transform(sp_raw.reshape(1, -1)).reshape(C, H, W)
        tm_norm = scaler_X_time.transform(tm_raw.reshape(1, -1)).reshape(-1)

        sp = torch.FloatTensor(sp_norm).unsqueeze(0).to(device)
        tm = torch.FloatTensor(tm_norm).unsqueeze(0).to(device)

        model_grids = {}
        with torch.no_grad():
            for name, model in loaded_models.items():
                pred, _, _ = model(sp, tm)
                grid = scaler_y.inverse_transform(pred.cpu().numpy().reshape(1, -1)).reshape(input_shape)
                model_grids[name] = grid

        # 1. 格网化匹配
        binned_df = match_jason_to_grid_multi(jason_df, model_grids, gold_valid_mask, lats, lons)
        if len(binned_df) == 0: continue

        binned_df['model_time'] = time_str

        # 2. 筛选有效区域
        valid_df = binned_df[binned_df['gold_validity'] > 0.5].copy()

        if len(valid_df) > 0:
            results_list.append(valid_df)

            # 绘图: 每20帧画一次
            if i % 20 == 0:
                print(f"   [Plotting] Overlay maps for {time_str}...")
                for model_name, grid in model_grids.items():
                    # 这里是关键：用新的 plot_overlay_map (pcolormesh)
                    plot_overlay_map(grid, valid_df, gold_valid_mask, lats, lons, time_str, plots_dir, model_name)

            if i % 50 == 0:
                print(f"Processed {time_str}: Found {len(valid_df)} points.")

    if not results_list:
        print("❌ 未找到匹配数据")
        return

    all_res = pd.concat(results_list, ignore_index=True)
    csv_path = os.path.join(eval_dir, 'jason_binned_stats_multi.csv')
    all_res.to_csv(csv_path, index=False)
    print(f"\n✅ 数据处理完成！CSV已保存: {csv_path} (Size: {len(all_res)} rows)")

    print("🎨 正在绘制全局散点图...")
    for model_name in loaded_models.keys():
        col_name = f'pred_{model_name}'
        plot_global_scatter(all_res, col_name, plots_dir, model_name)

    print(f"✅ 所有流程结束，结果保存在: {eval_dir}")


if __name__ == "__main__":
    evaluate_jason_pipeline()