import matplotlib

matplotlib.use('Agg')

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from torch.utils.data import Dataset, DataLoader

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.utils.common import setup_gpu
from src.models.fusion import TECFusionCNNModel

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.font_manager as fm

fm.fontManager.addfont('/home/ymli/Fonts/times.ttf')

# 后续的 rcParams 配置保持不变即可生效
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10
})

# =================================================================
# ⚙️ 配置区域
# =================================================================
OUTPUT_DIR = os.path.join(project_root, 'logs', 'model_comparison_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS_TO_COMPARE = [
    {
        'name': 'Fusion Model',
        'path': 'logs/exp4_fusion_no_attn/best_model.pth',
        'color': 'dodgerblue', 'style': '-', 'width': 1,
        'label': 'Fusion', 'zorder': 10
    },
    {
        'name': 'CNN-Only',
        'path': 'logs/exp2_cnn_only/best_model.pth',
        'color': 'red', 'style': '--', 'width': 1,
        'label': 'CNN', 'zorder': 9
    },
    {
        'name': 'BPNN-Only',
        'path': 'logs/exp3_bpnn_only/best_model.pth',
        'color': 'green', 'style': ':', 'width': 1,
        'label': 'BPNN', 'zorder': 9
    }
]

BG_COLORS = {
    'Train': '#FFFFFF',
    'Val': '#999999',
    'Test': '#99CCFF'
}


class EvaluatorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_dataset_once():
    base_path = Config.DATASET_PATH
    base, ext = os.path.splitext(base_path)

    split_paths = {
        'Train': f"{base}_train{ext}",
        'Val': f"{base}_val{ext}",
        'Test': f"{base}_test{ext}"
    }

    all_data = {}
    for split_name, path in split_paths.items():
        if os.path.exists(path):
            print(f"📦 Loading {split_name} set: {path}")
            try:
                raw = torch.load(path, weights_only=False)
            except TypeError:
                raw = torch.load(path)
            all_data[split_name] = raw
        else:
            print(f"⚠️ Missing {split_name} set: {path}")

    return all_data


def evaluate_single_model(model_conf, raw_data_dict, device):
    model_path = os.path.join(project_root, model_conf['path'])
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    print(f"\n🤖 Evaluating: {model_conf['name']} ...")

    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)

    config = ckpt['config']
    scaler_X_spatial = ckpt.get('scaler_X_spatial') or ckpt['scaler_X_spatial']
    scaler_X_time = ckpt.get('scaler_X_time') or ckpt['scaler_X_time']
    scaler_y = ckpt['scaler_y']

    model = TECFusionCNNModel(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    results = []

    for split_name, raw_samples in raw_data_dict.items():
        processed = []
        for s in raw_samples:
            sp = s[0].numpy()
            C, H, W = sp.shape
            sp_norm = scaler_X_spatial.transform(sp.reshape(1, -1)).reshape(C, H, W)
            tm = s[1].numpy().reshape(1, -1)
            tm_norm = scaler_X_time.transform(tm).reshape(-1)
            processed.append((torch.FloatTensor(sp_norm), torch.FloatTensor(tm_norm),
                              s[2], s[3], s[5], s[6] if len(s) > 6 else ""))

        loader = DataLoader(EvaluatorDataset(processed), batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                b_sp, b_tm, b_target, b_mit, b_gold, time_str = [x.to(device) if i < 5 else x for i, x in
                                                                 enumerate(batch)]
                time_str = time_str[0]

                out = model(b_sp, b_tm)
                if isinstance(out, tuple):
                    pred_raw = out[0]
                else:
                    pred_raw = out

                if pred_raw is None: continue

                pred = scaler_y.inverse_transform(pred_raw.cpu().numpy().reshape(1, -1)).reshape(
                    config['input_shape']).squeeze()
                target = b_target.cpu().numpy().reshape(config['input_shape']).squeeze()

                m_mit = b_mit.cpu().numpy().squeeze().astype(bool)
                m_gold = b_gold.cpu().numpy().squeeze().astype(bool)
                mask = m_mit & m_gold

                if mask.any():
                    rmse = np.sqrt(np.mean((target[mask] - pred[mask]) ** 2))
                    results.append({
                        'time': time_str,
                        'rmse': rmse,
                        'split': split_name,
                        'model': model_conf['name']
                    })

    return pd.DataFrame(results)


def plot_combined_chart(df):
    if df.empty: return

    print("\n🎨 Plotting Combined Chart (Single Plot)...")

    df['time_obj'] = pd.to_datetime(df['time'], format='%Y%m%d_%H%M%S', errors='coerce')
    if df['time_obj'].isna().any():
        df['time_obj'] = pd.to_datetime(df['time'], errors='coerce')

    df = df.sort_values('time_obj')
    pivot_df = df.pivot_table(index='time_obj', columns='model', values='rmse')
    # 应用3天滑动平均
    pivot_df = pivot_df.rolling(window='3D', center=True).mean()

    # =================================================================
    # 【修改】严格匹配论文 10 号 Times New Roman 字体
    # =================================================================
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    # 固定单栏绝对宽度
    a4_width_inch = 6.27
    # 既然图例移回内部，总高度可以从 3.6 英寸安全收紧回 3.0 英寸，使图表整体更紧凑
    fig_height = 3.0

    fig, ax_main = plt.subplots(1, 1, figsize=(a4_width_inch, fig_height))

    # 【关键修改】回收顶部物理空白：将 top 从 0.82 大幅放宽至 0.94，让数据绘图区长高
    plt.subplots_adjust(top=0.94, bottom=0.16, left=0.10, right=0.95)

    # 绘制背景 (7个周期 x 50天)
    print("   -> Painting background regions (Fixed 7 Cycles)...")
    cycle_base = pd.Timestamp("2023-01-01")
    cycle_rules = [
        ('Train', 35, BG_COLORS['Train']),
        ('Val', 10, BG_COLORS['Val']),
        ('Test', 5, BG_COLORS['Test'])
    ]
    added_labels = set()

    for i in range(7):
        curr_date = cycle_base + pd.Timedelta(days=i * 50)
        for split_name, days, color in cycle_rules:
            end_date = curr_date + pd.Timedelta(days=days)

            lbl = f"{split_name}" if split_name not in added_labels else None

            ax_main.axvspan(curr_date, end_date, color=color, alpha=0.4,
                            zorder=0, label=lbl, linewidth=0)

            if lbl: added_labels.add(split_name)
            curr_date = end_date

    ax_main.set_xlim(cycle_base, cycle_base + pd.Timedelta(days=350))

    # 主图：RMSE 时序
    print("   -> Plotting Main RMSE trends...")
    for conf in MODELS_TO_COMPARE:
        name = conf['name']
        if name not in pivot_df.columns: continue

        series = pivot_df[name]

        ax_main.plot(series.index, series,
                     color=conf['color'], linestyle=conf['style'], linewidth=conf['width'],
                     label=conf['label'], zorder=conf['zorder'])

    ax_main.set_ylabel('RMSE (TECU)')
    ax_main.set_ylim(3, 19)

    # 网格线保持
    ax_main.grid(True, linestyle='--', alpha=0.5, linewidth=0.3)

    # 图例代理对象处理
    handles, labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    for split_name in ['Train', 'Val', 'Test']:
        if split_name in by_label:
            fill_color = mcolors.to_rgba(BG_COLORS[split_name], alpha=0.4)
            by_label[split_name] = mpatches.Patch(
                facecolor=fill_color,
                edgecolor='black',
                linewidth=0.1,
                label=split_name
            )

    order_keys = [m['label'] for m in MODELS_TO_COMPARE] + ['Train', 'Val', 'Test']
    ordered_handles = [by_label[k] for k in order_keys if k in by_label]
    ordered_labels = [k for k in order_keys if k in by_label]

    # 【关键修改】图例移回内部左上角 (loc='upper left')
    # 采用 3 列平铺 (ncol=3) 压低高度，加白底半透明背景 (framealpha=0.9) 保证文字不被下方网格干扰
    ax_main.legend(ordered_handles, ordered_labels, loc='upper left',
                   ncol=3, framealpha=0.9, facecolor='white', edgecolor='none')

    # X 轴标准配置
    ax_main.set_xlabel('Time (Date)')
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    ax_main.tick_params(axis='both')

    save_path = os.path.join(OUTPUT_DIR, 'ablation_comparison_single.png')
    # 坚决不使用 bbox_inches='tight'，确保导出宽度锁死在 6.27 英寸不缩水
    plt.savefig(save_path, dpi=500, facecolor='white')
    plt.close()
    print(f"✅ Plot saved to: {save_path}")


def main():
    device = setup_gpu()

    raw_data = load_dataset_once()
    if not raw_data:
        print("❌ No datasets found!")
        return

    all_dfs = []
    for conf in MODELS_TO_COMPARE:
        df = evaluate_single_model(conf, raw_data, device)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        csv_path = os.path.join(OUTPUT_DIR, 'ablation_metrics_raw.csv')
        full_df.to_csv(csv_path, index=False)
        print(f"💾 Metrics saved to {csv_path}")

        plot_combined_chart(full_df)
    else:
        print("❌ No evaluation results generated.")


if __name__ == "__main__":
    main()