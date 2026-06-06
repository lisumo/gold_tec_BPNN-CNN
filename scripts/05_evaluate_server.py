import matplotlib

# 1. 强制设置 Agg 后端
matplotlib.use('Agg')

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Cartopy 导入 (可选) ===
try:
    import cartopy.crs as ccrs
    from cartopy.feature import NaturalEarthFeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.utils.common import setup_gpu
from src.models.fusion import TECFusionCNNModel
from src.utils.options import parse_args

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class EvaluatorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # 0:sp, 1:tm, 2:tgt, 3:mit, 4:aux, 5:gold
        spatial = item[0]
        time_feat = item[1]
        target = item[2]
        mit_mask = item[3]
        gold_mask = item[5]
        time_str = item[6] if len(item) > 6 else ""
        return spatial, time_feat, target, mit_mask, gold_mask, time_str


def parse_region_str(region_str):
    if not region_str: return None
    try:
        vals = list(map(float, region_str.split(',')))
        return {'lon_min': vals[0], 'lon_max': vals[1], 'lat_min': vals[2], 'lat_max': vals[3]}
    except:
        return None


def add_map_features(ax):
    """辅助函数：添加地图要素"""
    if not HAS_CARTOPY: return
    coastline = NaturalEarthFeature('physical', 'coastline', '50m', edgecolor='black', facecolor='none')
    ax.add_feature(coastline, linewidth=0.5)
    borders = NaturalEarthFeature('cultural', 'admin_0_countries', '50m', edgecolor='black', facecolor='none')
    ax.add_feature(borders, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


def plot_spatial_analysis(save_dir, time_str, map_true, map_pred, map_cnn, map_bpnn,
                          mask_mit, mask_gold, mask_inter, lon_grid, lat_grid):
    """
    绘制空间分析图（拆分单独保存，适配A4纸一行5个图）
    文件名格式：{time}_{type}_{model}.png
    """
    if not HAS_CARTOPY:
        return

    # 准备可视化数据（掩码区域填NaN）
    vis_true = np.where(mask_mit, map_true, np.nan)
    vis_pred = np.where(mask_gold, map_pred, np.nan)
    vis_cnn = np.where(mask_gold, map_cnn, np.nan) if map_cnn is not None else np.full_like(vis_pred, np.nan)
    vis_bpnn = np.where(mask_gold, map_bpnn, np.nan) if map_bpnn is not None else np.full_like(vis_pred, np.nan)
    vis_diff = np.where(mask_inter, map_pred - map_true, np.nan)

    # 动态计算色标范围
    vmin = np.nanmin(vis_true) if not np.all(np.isnan(vis_true)) else 0
    vmax = np.nanmax(vis_true) if not np.all(np.isnan(vis_true)) else 80
    diff_max = np.nanmax(np.abs(vis_diff)) if np.any(~np.isnan(vis_diff)) else 5.0

    # 图尺寸：A4宽8.27in - 边距1in = 7.27in，除以5 = 1.45in
    fig_width = 1.6
    fig_height = 1.45

    # 定义要绘制的图列表：(数据, cmap, vmin, vmax, 文件名后缀)
    plots = [
        (vis_true, 'viridis', vmin, vmax, 'truth_MIT'),
        (vis_bpnn, 'viridis', vmin, vmax, 'pred_BPNN'),
        (vis_cnn, 'viridis', vmin, vmax, 'pred_CNN'),
        (vis_pred, 'viridis', vmin, vmax, 'pred_Fusion'),
        (vis_diff, 'RdBu_r', -diff_max, diff_max, 'diff_Fusion')
    ]

    # 安全文件名
    safe_time = time_str.replace(':', '-').replace(' ', '_').replace('-', '')

    for data, cmap, v_min, v_max, suffix in plots:
        # 跳过无效数据（如BPNN/CNN为None时）
        if np.all(np.isnan(data)):
            continue

        fig, ax = plt.subplots(figsize=(fig_width, fig_height),
                               subplot_kw={'projection': ccrs.PlateCarree()})

        # 地图要素：无填充，仅轮廓线
        coastline = NaturalEarthFeature('physical', 'coastline', '50m',
                                        edgecolor='black', facecolor='none', linewidth=0.3)
        ax.add_feature(coastline)
        borders = NaturalEarthFeature('cultural', 'admin_0_countries', '50m',
                                      edgecolor='black', facecolor='none', linewidth=0.2)
        ax.add_feature(borders)

        # 网格线字体5pt
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4,
                          linewidth=0.3, color='gray')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        # 绘制数据
        im = ax.pcolormesh(lon_grid, lat_grid, data, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=v_min, vmax=v_max, shading='nearest')

        # 确保坐标轴区域为正方形
        ax.set_aspect('equal', adjustable='box')

        # 设置范围
        ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()],
                      crs=ccrs.PlateCarree())

        # 【修复】使用plt.colorbar的shrink和pad参数，不使用make_axes_locatable
        cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.03)
        #cbar.set_label('TEC' if 'diff' not in suffix else 'ΔTEC', fontsize=5)
        cbar.ax.tick_params(labelsize=6)

        # 紧凑布局
        plt.tight_layout(pad=0.1)

        # 保存
        filename = f"{safe_time}_{suffix}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()

    print(f"   Saved 5 spatial maps to {save_dir} for {time_str}")


def evaluate_pipeline():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = setup_gpu()

    # === 1. 模型加载 ===
    log_dir = os.path.join(project_root, 'logs', args.exp_name)
    model_path = os.path.join(log_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        logs_root = os.path.join(project_root, 'logs')
        if os.path.exists(logs_root):
            exps = sorted([os.path.join(logs_root, d) for d in os.listdir(logs_root) if
                           os.path.isdir(os.path.join(logs_root, d))], key=os.path.getmtime)
            if exps:
                model_path = os.path.join(exps[-1], 'best_model.pth')
                log_dir = exps[-1]
                print(f"Redirecting to latest experiment: {model_path}")

    eval_dir = os.path.join(log_dir, 'evaluation_results')
    os.makedirs(eval_dir, exist_ok=True)

    print(f"📂 Loading model: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint['config']

    # 区域配置
    target_region = None
    if args.mask_region:
        target_region = parse_region_str(args.mask_region)
    if target_region is None and model_config.get('enable_land_masking'):
        target_region = model_config.get('land_mask_region')
    if target_region is None and getattr(Config, 'ENABLE_LAND_MASKING', False):
        target_region = getattr(Config, 'LAND_MASK_REGION', None)

    if target_region:
        Config.ENABLE_LAND_MASKING = True
        Config.LAND_MASK_REGION = target_region
    else:
        Config.ENABLE_LAND_MASKING = False

    scaler_y = checkpoint['scaler_y']
    scaler_X_spatial = checkpoint.get('scaler_X_spatial') or checkpoint['scaler_X_spatial']
    scaler_X_time = checkpoint.get('scaler_X_time') or checkpoint['scaler_X_time']

    model = TECFusionCNNModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # === 2. 准备数据集 ===
    base_path = args.dataset_path if args.dataset_path else Config.DATASET_PATH
    base, ext = os.path.splitext(base_path)

    split_paths = {
        'Train': f"{base}_train{ext}",
        'Val': f"{base}_val{ext}",
        'Test': f"{base}_test{ext}"
    }

    # 准备网格坐标 (用于绘图)
    lon_arr = np.linspace(Config.MASK_LON_MIN, Config.MASK_LON_MAX, model_config['input_shape'][1])
    lat_arr = np.linspace(Config.MASK_LAT_MIN, Config.MASK_LAT_MAX, model_config['input_shape'][0])
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    all_metrics_list = []

    # === 3. 循环处理 ===
    for split_name, split_path in split_paths.items():
        if not os.path.exists(split_path): continue
        print(f"\n🚀 Processing {split_name} Set: {split_path}")

        try:
            raw_samples = torch.load(split_path, weights_only=False)
        except TypeError:
            raw_samples = torch.load(split_path)

        processed_samples = []
        for s in raw_samples:
            sp = s[0].numpy()
            C, H, W = sp.shape
            sp_flat = sp.reshape(1, -1)
            sp_norm = scaler_X_spatial.transform(sp_flat).reshape(C, H, W)
            tm = s[1].numpy().reshape(1, -1)
            tm_norm = scaler_X_time.transform(tm).reshape(-1)
            new_sample = [torch.FloatTensor(sp_norm), torch.FloatTensor(tm_norm), *s[2:]]
            processed_samples.append(tuple(new_sample))

        loader = DataLoader(EvaluatorDataset(processed_samples), batch_size=1, shuffle=False)

        split_true = []
        split_pred = []

        # 如果是 Test 集，创建专门的图片目录
        test_maps_dir = os.path.join(eval_dir, 'test_maps')
        if split_name == 'Test':
            os.makedirs(test_maps_dir, exist_ok=True)
            print(f"   📷 Spatial maps for Test set will be saved to: {test_maps_dir}")

        with torch.no_grad():
            for i, batch in enumerate(loader):
                b_sp, b_tm, b_target, b_mit_mask, b_gold_mask = [x.to(device) for x in batch[:5]]
                time_str = batch[5][0] if len(batch) > 5 else f"S{i}"

                final, cnn, bpnn = model(b_sp, b_tm)

                def inverse(t):
                    if t is None: return None
                    arr = t.cpu().numpy().reshape(1, -1)
                    return scaler_y.inverse_transform(arr).reshape(model_config['input_shape'])

                map_pred = inverse(final)
                map_cnn = inverse(cnn)
                map_bpnn = inverse(bpnn)
                map_true = b_target.cpu().numpy().reshape(model_config['input_shape'])
                mask_mit = b_mit_mask.cpu().numpy().reshape(model_config['input_shape']).astype(bool)
                mask_gold = b_gold_mask.cpu().numpy().reshape(model_config['input_shape']).astype(bool)
                mask_inter = mask_mit & mask_gold
                # [新增] 如果指定了目标区域，应用区域掩膜进行筛选
                if target_region:
                    # target_region = {'lon_min':..., 'lon_max':..., 'lat_min':..., 'lat_max':...}
                    # 利用之前准备好的网格 lon_grid, lat_grid (注意它们已经在前面定义过了)
                    region_mask = (lon_grid >= target_region['lon_min']) & \
                                  (lon_grid <= target_region['lon_max']) & \
                                  (lat_grid >= target_region['lat_min']) & \
                                  (lat_grid <= target_region['lat_max'])
                    # 将区域外的点在 mask_inter 中设为 False
                    mask_inter = mask_inter & region_mask

                # --- 1. 指标计算 ---
                if np.any(mask_inter):
                    valid_true = map_true[mask_inter]
                    valid_pred = map_pred[mask_inter]
                    split_true.append(valid_true)
                    split_pred.append(valid_pred)

                    mse_val = np.mean((valid_true - valid_pred) ** 2)
                    rmse_val = np.sqrt(mse_val)

                    cnn_rmse = np.nan
                    if map_cnn is not None:
                        cnn_rmse = np.sqrt(np.mean((valid_true - map_cnn[mask_inter]) ** 2))

                    bpnn_rmse = np.nan
                    if map_bpnn is not None:
                        bpnn_rmse = np.sqrt(np.mean((valid_true - map_bpnn[mask_inter]) ** 2))

                    all_metrics_list.append({
                        'time': time_str,
                        'split': split_name,
                        'Fusion': rmse_val,
                        'CNN': cnn_rmse,
                        'BPNN': bpnn_rmse
                    })

                # --- 2. 空间绘图 (仅针对 Test 集) ---
                if split_name == 'Test':
                    # 将时间字符串转换为安全的文件名
                    safe_name = time_str.replace(':', '-').replace(' ', '_')
                    save_path = os.path.join(test_maps_dir, f"Analysis_{safe_name}.png")

                    # 调用绘图函数
                    plot_spatial_analysis(
                        test_maps_dir, time_str,  # 传入目录而非文件路径
                        map_true, map_pred, map_cnn, map_bpnn,
                        mask_mit, mask_gold, mask_inter,
                        lon_grid, lat_grid
                    )

                    # 打印进度 (每50张)
                    if i % 50 == 0:
                        print(f"   -> Saved map {i}/{len(loader)}: {save_path}")

        if split_true:
            rmse = np.sqrt(mean_squared_error(np.concatenate(split_true), np.concatenate(split_pred)))
            print(f"📊 {split_name} Global RMSE: {rmse:.4f} TECU")

    # === 4. 绘制全时段时序图 (含背景色) ===
    if all_metrics_list:
        print("\n🎨 Plotting All-Data Time Series with Background Colors...")
        df = pd.DataFrame(all_metrics_list)
        df['time_obj'] = pd.to_datetime(df['time'], format='%Y%m%d_%H%M%S', errors='coerce')
        if df['time_obj'].isna().any(): df['time_obj'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.sort_values('time_obj').reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df['time_obj'], df['Fusion'], label='Fusion Model', color='#D62728', linewidth=1.5, zorder=10)
        if not df['CNN'].isna().all():
            ax.plot(df['time_obj'], df['CNN'], label='CNN Branch', color='#2CA02C', linewidth=1, linestyle='--',
                    alpha=0.8, zorder=9)
        if not df['BPNN'].isna().all():
            ax.plot(df['time_obj'], df['BPNN'], label='BPNN Branch', color='#1F77B4', linewidth=1, linestyle=':',
                    alpha=0.8, zorder=9)

        df['group'] = (df['split'] != df['split'].shift()).cumsum()

        bg_rules = [
            ('Train', 35, '#E6F5C9'),
            ('Val', 10, '#FFF2AE'),
            ('Test', 5, '#F4CAE4')
        ]

        cycle_base = pd.Timestamp("2023-01-01")
        used_labels = set()

        for i in range(7):  # 7个循环
            curr_date = cycle_base + pd.Timedelta(days=i * 50)
            for split_name, days, color in bg_rules:
                end_date = curr_date + pd.Timedelta(days=days)
                lbl = f"{split_name} Set" if split_name not in used_labels else None
                ax.axvspan(curr_date, end_date, color=color, alpha=0.5,
                           label=lbl, zorder=0, linewidth=0)
                if lbl: used_labels.add(split_name)

                curr_date = end_date
        # [重要] 强制设置 X 轴范围，保证背景完整
        ax.set_xlim(cycle_base, cycle_base + pd.Timedelta(days=350))

        ax.set_xlabel('Time (Date)', fontsize=12)
        ax.set_ylabel('RMSE (TECU)', fontsize=12)
        ax.set_title('RMSE Time Series Analysis across Train/Val/Test Sets', fontsize=14, pad=15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        ax.grid(True, linestyle='--', alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        order = ['Fusion Model', 'CNN Branch', 'BPNN Branch', 'Train Set', 'Val Set', 'Test Set']
        ordered_handles = [by_label[k] for k in order if k in by_label]
        ordered_labels = [k for k in order if k in by_label]
        ax.legend(ordered_handles, ordered_labels, loc='upper right', framealpha=0.9, shadow=True)

        plt.tight_layout()
        save_path = os.path.join(eval_dir, 'full_dataset_rmse_timeseries.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Full time series plot saved to: {save_path}")

    else:
        print("❌ No data collected for plotting.")


if __name__ == "__main__":
    evaluate_pipeline()