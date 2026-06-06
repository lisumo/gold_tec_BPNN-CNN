import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyIGRF14 as IGRF
from matplotlib.gridspec import GridSpec


def plot_igrf_manual_layout(year=2023, alt=350):
    """
    IGRF 地磁分量 - 科研绘图规范版
    画布宽度固定为A4纸可插入最大宽度，字号固定为8pt
    """
    print(f"正在计算 {year} 年 {alt} km 高度的全量地磁数据...")

    # 1. 准备数据
    lat = np.linspace(-90, 90, 91)
    lon = np.linspace(-180, 180, 181)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    data_map = {}
    keys = ['D', 'I', 'H', 'X', 'Y', 'Z', 'F']
    for k in keys:
        data_map[k] = np.zeros_like(lat_grid)

    rows, cols = lat_grid.shape
    for r in range(rows):
        for c in range(cols):
            d, i_val, h, x, y, z, f = IGRF.igrf_value(
                lat=lat_grid[r, c], lon=lon_grid[r, c],
                alt=alt, year=year
            )
            data_map['D'][r, c] = d
            data_map['I'][r, c] = i_val
            data_map['H'][r, c] = h
            data_map['X'][r, c] = x
            data_map['Y'][r, c] = y
            data_map['Z'][r, c] = z
            data_map['F'][r, c] = f

    print("计算完成，开始精细绘图...")

    # 2. 科研绘图规范：A4纸宽度 + 固定8号字
    a4_width_inch = 6.14
    fig_height = 2.0

    fig = plt.figure(figsize=(a4_width_inch, fig_height))

    # 使用GridSpec，为右侧色标留出空间
    gs = GridSpec(2, 4, figure=fig,
                  left=0.08, right=1.2,  # 右侧留出空间给色标
                  top=0.95, bottom=0.05,
                  wspace=0.4, hspace=0.01)

    # 3. 定义绘图配置
    configs = [
        ('D', 'Declination (D)', '°', 'winter', True, [-180, -90, 0, 90, 180]),
        ('I', 'Inclination (I)', '°', 'winter', True, [-90, -45, 0, 45, 90]),
        ('H', 'Horizontal Intensity (H)', 'nT', 'winter', False, [0, 10000, 20000, 30000]),
        ('X', 'North Component (X)', 'nT', 'winter', True, [-20000, 0, 20000]),
        ('Y', 'East Component (Y)', 'nT', 'winter', True, [-15000, 0, 15000]),
        ('Z', 'Vertical Component (Z)', 'nT', 'winter', True, [-60000, 0, 60000])
    ]

    # 4. 循环绘图
    for i, config in enumerate(configs):
        row = i // 3
        col = i % 3

        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())

        show_yticks = (col == 0)
        show_xticks = (row == 1)

        _plot_single_ax(fig, ax, lon, lat, data_map, config, show_yticks, show_xticks)

    save_path = 'IGRF_Manual_Perfect.png'
    plt.savefig(save_path, dpi=500)
    print(f"✅ 完美排版图已保存至: {save_path}")
    plt.show()


def _plot_single_ax(fig, ax, lon, lat, data_map, config, show_yticks=True, show_xticks=True):
    """通用绘图函数 - 色标放右侧，标题正常大小"""
    key, title, unit, cmap, symmetric, cbar_ticks = config
    data = data_map[key]

    # 底图
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.3, linewidth=0.2)

    # 确定色阶
    if symmetric:
        vmax = np.max(np.abs(data))
        vmin = -vmax
        levels = np.linspace(vmin, vmax, 31)
    else:
        vmin = np.min(data)
        vmax = np.max(data)
        levels = np.linspace(vmin, vmax, 31)

    # 填色
    cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap,
                     transform=ccrs.PlateCarree(), extend='both')

    # 标注地理赤道
    ax.plot([-180, 180], [0, 0], 'k-', lw=0.3, alpha=0.3, transform=ccrs.PlateCarree())

    # 标注磁赤道
    if key in ['I', 'Z']:
        ax.contour(lon, lat, data, levels=[0], colors='white', linewidths=0.8,
                   linestyles='--', transform=ccrs.PlateCarree())
        ax.text(-160, -25, 'Dip Eq.', color='white', fontsize=5,
                fontweight='bold', transform=ccrs.PlateCarree())

    # 标题 - 正常大小，不加粗
    ax.set_title(title, fontsize=8, pad=3)

    # 设置坐标范围
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # 设置刻度
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-90, -45, 0, 45, 90])

    # 控制刻度标签显示
    if show_yticks:
        ax.set_yticklabels(['90°S', '45°S', '0°', '45°N', '90°N'], fontsize=6)
    else:
        ax.set_yticklabels([])
        ax.tick_params(left=False)

    if show_xticks:
        ax.set_xticklabels(['180°', '90°', '0°', '90°', '180°'], fontsize=6)
    else:
        ax.set_xticklabels([])
        ax.tick_params(bottom=False)

    # 色标 - 放在子图右侧
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax = inset_axes(ax, width="4%", height="100%", loc='lower left',
                     bbox_to_anchor=(1.05, 0, 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)

    cbar = fig.colorbar(cf, cax=cax, orientation='vertical', ticks=cbar_ticks)
    cbar.set_label(unit, fontsize=7, rotation=0, labelpad=2)
    cbar.ax.tick_params(labelsize=6)

    # 简化大数字显示
    if key in ['H', 'F', 'X', 'Y', 'Z']:
        cbar.ax.set_yticklabels([f'{x/1000:.0f}k' if abs(x) >= 1000 else f'{x:.0f}'
                                  for x in cbar_ticks])


if __name__ == "__main__":
    plot_igrf_manual_layout()