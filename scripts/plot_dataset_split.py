import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import numpy as np

def draw_periodic_split_diagram_v4():
    # 参数设置
    total_days = 350
    window_size = 50
    train_days = 35
    val_days = 10
    test_days = 5
    n_cycles = total_days // window_size

    # 配色
    color_train = '#4e79a7'
    color_val = '#f28e2b'
    color_test = '#59a14f'
    color_bg = '#f0f0f0'

    # 全局注入标准论文衬线字体及 10 号字
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    # 严格限定单栏物理画布尺寸
    a4_width_inch = 6.27
    fig_height = 3.2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(a4_width_inch, fig_height),
                                   gridspec_kw={'height_ratios': [1, 1.1]})

    # 布局微调
    plt.subplots_adjust(hspace=0.70, top=0.92, bottom=0.25, left=0.06, right=0.94)

    # ================= (a) 全局视图 =================
    ax1_rect_h = 0.03
    ax1_rect_y = 0.02
    ax1_ylim = 0.11

    ax1.set_xlim(0, total_days)
    ax1.set_ylim(0, ax1_ylim)
    ax1.set_yticks([])
    ax1.set_xlabel('Day of Year (DOY)', labelpad=4)

    # 【精准微调 1】引入 y=0.68，将图名下压至隐形真空区，紧贴 Cycle 文字上方
    ax1.set_title('(a) Global Periodic Splitting Strategy (Total 350 Days)',
                  loc='left', fontweight='bold', y=0.68)

    ax1.add_patch(patches.Rectangle((0, ax1_rect_y), total_days, ax1_rect_h,
                                    color=color_bg, alpha=0.5))

    for i in range(n_cycles):
        start = i * window_size
        # Training
        ax1.add_patch(patches.Rectangle((start, ax1_rect_y), train_days, ax1_rect_h,
                                        color=color_train, alpha=0.9))
        # Validation 【已修正参数错位问题】
        ax1.add_patch(patches.Rectangle((start + train_days, ax1_rect_y), val_days,
                                        ax1_rect_h, color=color_val, alpha=0.9))
        # Test
        ax1.add_patch(patches.Rectangle((start + train_days + val_days, ax1_rect_y),
                                        test_days, ax1_rect_h, color=color_test, alpha=0.9))

        text_y = ax1_rect_y + ax1_rect_h + 0.005
        ax1.text(start + window_size / 2, text_y, f'Cycle {i + 1}', ha='center', va='bottom')

        if i < n_cycles:
            ax1.axvline(x=start + window_size, color='white', linewidth=0.8,
                        ymin=ax1_rect_y / ax1_ylim,
                        ymax=(ax1_rect_y + ax1_rect_h) / ax1_ylim)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # ================= (b) 局部放大视图 =================
    ax2_rect_h = 0.04
    ax2_rect_y = 0.01
    ax2_ylim = 0.075

    ax2.set_xlim(0, window_size)
    ax2.set_ylim(0, ax2_ylim)
    ax2.set_yticks([])
    ax2.set_xlabel('Days within a Window', labelpad=4)

    # 【精准微调 2】引入 y=0.78，消除图名 (b) 与下方大色块之间的多余悬空空隙
    ax2.set_title('(b) Detailed Split within a Single Window (50 Days)',
                  loc='left', fontweight='bold', y=0.78)

    text_center_y = ax2_rect_y + ax2_rect_h / 2

    # Train
    ax2.add_patch(patches.Rectangle((0, ax2_rect_y), train_days, ax2_rect_h,
                                    color=color_train, alpha=0.8))
    ax2.text(train_days / 2, text_center_y, f'Training\n({train_days} days)',
             ha='center', va='center', color='white')

    # Val
    ax2.add_patch(patches.Rectangle((train_days, ax2_rect_y), val_days, ax2_rect_h,
                                    color=color_val, alpha=0.8))
    ax2.text(train_days + val_days / 2, text_center_y, f'Validation\n({val_days} days)',
             ha='center', va='center', color='white')

    # Test
    ax2.add_patch(patches.Rectangle((train_days + val_days, ax2_rect_y), test_days,
                                    ax2_rect_h, color=color_test, alpha=0.8))
    ax2.text(train_days + val_days + test_days / 2, text_center_y, f'Test\n({test_days} days)',
             ha='center', va='center', color='white')

    # 连接线
    con1 = ConnectionPatch(xyA=(0, ax1_rect_y), xyB=(0, ax2_rect_y + ax2_rect_h),
                           coordsA="data", coordsB="data",
                           axesA=ax1, axesB=ax2, color="gray", linestyle="--",
                           alpha=0.5, linewidth=0.8)
    con2 = ConnectionPatch(xyA=(window_size, ax1_rect_y), xyB=(window_size, ax2_rect_y + ax2_rect_h),
                           coordsA="data", coordsB="data",
                           axesA=ax1, axesB=ax2, color="gray", linestyle="--",
                           alpha=0.5, linewidth=0.8)
    ax2.add_artist(con1)
    ax2.add_artist(con2)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # ================= 3. 图例 =================
    legend_elements = [
        patches.Patch(facecolor=color_train, edgecolor='none', label='Training'),
        patches.Patch(facecolor=color_val, edgecolor='none', label='Validation'),
        patches.Patch(facecolor=color_test, edgecolor='none', label='Test')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.savefig('dataset_split_scheme_v4.png', dpi=300, facecolor='white')
    plt.close()
    print("✅ 间距微调与参数修正完成，已成功生成图片。")

if __name__ == "__main__":
    draw_periodic_split_diagram_v4()