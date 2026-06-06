import matplotlib

matplotlib.use('Agg')

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib.font_manager as fm

fm.fontManager.addfont('/home/ymli/Fonts/times.ttf')

# 后续的 rcParams 配置保持不变即可生效
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10
})

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# =================================================================
# ⚙️ 配置区域（【重大修改】全面重构为 SCI 高级学术哑光色系，杜绝大红大绿）
# =================================================================
CSV_PATH = "logs/jason_comparison_results/jason_binned_stats_multi.csv"

MODEL_STYLES = {#4e79a7
    'fusion': {'color': '#e2745e', 'style': '-', 'marker': '.', 'label': 'Fusion'},
    'cnn': {'color': '#f2ab6a', 'style': '--', 'marker': '+', 'label': 'CNN'},
    'bpnn': {'color': '#4c9ac9', 'style': ':', 'marker': 'x', 'label': 'BPNN'}
}


def process_and_plot():
    full_csv_path = os.path.join(project_root, CSV_PATH)
    output_dir = os.path.dirname(full_csv_path)

    if not os.path.exists(full_csv_path):
        print(f"❌ 错误: 文件不存在 -> {full_csv_path}")
        return

    print(f"📖 正在读取: {full_csv_path} ...")
    df = pd.read_csv(full_csv_path)

    if df.empty:
        print("❌ CSV 为空")
        return

    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    models_found = [c.replace('pred_', '') for c in pred_cols]
    print(f"🔍 检测到模型: {models_found}")

    # 数据过滤
    time_counts = df['model_time'].value_counts()
    valid_times = time_counts[time_counts >= 10].index
    n_dropped = len(time_counts) - len(valid_times)

    if n_dropped > 0:
        print(f"📉 过滤: 剔除 {n_dropped} 个数据量少于 10 的时刻")
        df = df[df['model_time'].isin(valid_times)].copy()

    df['time_obj'] = pd.to_datetime(df['model_time'], format='%Y%m%d_%H%M%S', errors='coerce')
    if df['time_obj'].isna().any():
        df.loc[df['time_obj'].isna(), 'time_obj'] = pd.to_datetime(df.loc[df['time_obj'].isna(), 'model_time'],
                                                                   errors='coerce')
    df = df.dropna(subset=['time_obj'])

    # 计算全局统计指标
    print("\n" + "=" * 65)
    print(f"{'Model Name':<15} | {'Global RMSE':<12} | {'Global Bias':<12} | {'Global STD':<12}")
    print("-" * 65)

    stats_results = {}
    for model_name in models_found:
        col_name = f'pred_{model_name}'
        diff = df[col_name] - df['jason_tec']
        g_rmse = np.sqrt((diff ** 2).mean())
        g_bias = diff.mean()
        g_std = diff.std(ddof=0)
        print(f"{model_name:<15} | {g_rmse:<12.4f} | {g_bias:<12.4f} | {g_std:<12.4f}")

        sq_diff = (df[col_name] - df['jason_tec']) ** 2
        daily_rmse = np.sqrt(sq_diff.groupby(df['model_time']).mean())
        stats_results[model_name] = daily_rmse
    print("=" * 65 + "\n")

    plot_df = pd.DataFrame(stats_results).reset_index()
    plot_df['time_obj'] = pd.to_datetime(plot_df['model_time'], format='%Y%m%d_%H%M%S', errors='coerce')
    if plot_df['time_obj'].isna().any():
        plot_df['time_obj'] = pd.to_datetime(plot_df['model_time'], errors='coerce')
    plot_df = plot_df.sort_values('time_obj').dropna(subset=['time_obj'])

    # 自动断轴分段
    time_diff = plot_df['time_obj'].diff()
    gap_mask = time_diff > pd.Timedelta(days=10)
    plot_df['segment_id'] = gap_mask.cumsum()

    daily_segments = [d for _, d in plot_df.groupby('segment_id')]
    n_segments = len(daily_segments)

    segment_days = []
    segment_ranges = []
    for seg in daily_segments:
        start = seg['time_obj'].min()
        end = seg['time_obj'].max()
        days = (end - start).days
        segment_days.append(max(days, 5))
        segment_ranges.append((start, end))

    width_ratios = np.array(segment_days) / np.sum(segment_days)

    # 全局注入标准论文 10 号 Times New Roman 字体
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    a4_width_inch = 6.27
    fig_height = 3.0

    # =================================================================
    # 🎨 图表 1: 折线图 (Line Chart) 同步更新颜色
    # =================================================================
    print(f"🎨 正在绘制折线图...")

    fig, axes = plt.subplots(1, n_segments, sharey=True, figsize=(a4_width_inch, fig_height),
                             gridspec_kw={'width_ratios': width_ratios})
    if n_segments == 1: axes = [axes]

    plt.subplots_adjust(bottom=0.18, top=0.92, left=0.10, right=0.95, wspace=0.06)

    legend_handles = []
    legend_labels = []

    for i, (ax, seg_df) in enumerate(zip(axes, daily_segments)):
        for model_name in models_found:
            style = MODEL_STYLES.get(model_name, {'color': 'gray', 'style': ':', 'marker': '', 'label': model_name})

            line, = ax.plot(seg_df['time_obj'], seg_df[model_name],
                            color=style['color'], linestyle=style['style'], linewidth=1.0,
                            marker=style['marker'], markersize=3, alpha=0.9,
                            label=style['label'])

            if i == 0:
                legend_handles.append(line)
                legend_labels.append(style['label'])

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.3)

        d = .02
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=0.8)
        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        if i < n_segments - 1:
            ax.spines['right'].set_visible(False)
            ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    axes[0].set_ylabel('RMSE (TECU)')
    fig.text(0.5, 0.02, 'Time (Date)', ha='center')

    axes[-1].legend(handles=legend_handles, labels=legend_labels,
                    loc='upper right', ncol=3, frameon=False)

    save_path_line = os.path.join(output_dir, 'jason_multi_model_comparison.png')
    plt.savefig(save_path_line, dpi=500, facecolor='white')
    plt.close()
    print(f"✅ 折线图已保存至: {save_path_line}")

    # =================================================================
    # 📊 图表 2: 柱状图 (Bar Chart) 重构
    # =================================================================
    print(f"📊 正在绘制柱状图 (Segment Pooled RMSE)...")

    fig_bar, axes_bar = plt.subplots(1, n_segments, sharey=True, figsize=(a4_width_inch, fig_height))
    if n_segments == 1: axes_bar = [axes_bar]

    plt.subplots_adjust(bottom=0.22, top=0.92, left=0.10, right=0.95, wspace=0.06)

    legend_handles_bar = []
    legend_labels_bar = []
    all_rmse_values = []

    for i, (ax, time_range) in enumerate(zip(axes_bar, segment_ranges)):
        start_date, end_date = time_range
        mask = (df['time_obj'] >= start_date) & (df['time_obj'] <= end_date + pd.Timedelta(days=1))
        seg_raw_data = df[mask]

        if seg_raw_data.empty: continue

        center_pos = 0.5
        total_bar_width = 0.75
        n_models = len(models_found)
        single_bar_width = total_bar_width / n_models
        offsets = np.arange(n_models) * single_bar_width - (total_bar_width - single_bar_width) / 2

        for idx, model_name in enumerate(models_found):
            col_name = f'pred_{model_name}'
            rmse_val = np.sqrt(((seg_raw_data[col_name] - seg_raw_data['jason_tec']) ** 2).mean())
            all_rmse_values.append(rmse_val)

            style = MODEL_STYLES.get(model_name, {'color': 'gray', 'label': model_name})

            # 【关键修改】设置 edgecolor='none', linewidth=0 彻底剥离边框
            bars = ax.bar(center_pos + offsets[idx], rmse_val,
                          width=single_bar_width,
                          color=style['color'], alpha=0.9,
                          label=style['label'],
                          edgecolor='none', linewidth=0)

            # 【关键修改】数值标注独立声明字号为 8 号 (fontsize=8)，杜绝文字大面积碰撞打架
            ax.text(center_pos + offsets[idx], rmse_val + 0.35, f"{rmse_val:.1f}",
                    ha='center', va='bottom', fontsize=8, transform=ax.transData)

            if i == 0:
                legend_handles_bar.append(bars)
                legend_labels_bar.append(style['label'])

        ax.set_xticks([])
        time_label = f"{start_date.strftime('%m-%d')}\n~ {end_date.strftime('%m-%d')}"
        ax.set_xlabel(time_label, labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y', linewidth=0.3)
        ax.set_xlim(0, 1)

        # 断轴装饰
        d = .02
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=0.8)
        if i > 0:
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        if i < n_segments - 1:
            ax.spines['right'].set_visible(False)
            ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    axes_bar[0].set_ylabel('Period RMSE (TECU)')
    axes_bar[0].tick_params(axis='y')

    if all_rmse_values:
        y_max = max(all_rmse_values)
        axes_bar[0].set_ylim(0, y_max * 1.4)

    axes_bar[-1].legend(handles=legend_handles_bar, labels=legend_labels_bar,
                        loc='upper right', ncol=3, frameon=False)

    save_path_bar = os.path.join(output_dir, 'jason_multi_model_comparison_bar.png')
    plt.savefig(save_path_bar, dpi=500, facecolor='white')
    plt.close()
    print(f"✅ 柱状图已保存至: {save_path_bar}")


if __name__ == "__main__":
    process_and_plot()