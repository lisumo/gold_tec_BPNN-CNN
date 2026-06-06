import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ppgnss import gnss_utils
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature

def add_map_features(ax):
    coastline = NaturalEarthFeature(
        category='physical',
        name='coastline',
        scale='10m',
        edgecolor='black',
        facecolor='none'
    )
    ax.add_feature(coastline, linewidth=0.3)

    borders = NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries',
        scale='10m',
        edgecolor='black',
        facecolor='none'
    )
    ax.add_feature(borders, linewidth=0.3)

def plot_mit_global_tec(data_dir, year, target_time, output_dir="mit_tec_figures"):
    """
    绘制指定时刻MIT的全球TEC数据

    参数:
        data_dir: GIM数据根目录
        year: 年份
        target_time: 目标时刻 (datetime64格式)
        output_dir: 图像输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义MIT数据文件路径
    mit_dir = os.path.join(data_dir, "TEC")
    mit_file = os.path.join(mit_dir, f"mitg{year:04d}_{year + 1:04d}.obj")

    # 加载MIT数据
    xr_mit = gnss_utils.loadobject(mit_file)

    # 选择目标时刻的数据（考虑MIT数据2.5分钟的时间偏移）
    mit_time = target_time + np.timedelta64(150, "s")  # 2.5分钟=150秒
    xr_mit_sel = xr_mit.sel(time=mit_time, method="nearest")

    # 计算经度偏移量（修复时间差计算方式）
    # 以当天0点为基准计算小时数
    year_start = np.datetime64(f"{year}-01-01")
    # 计算与年初的天数差（使用astype获取总秒数后转换为天数）
    days_since_start = (target_time - year_start).astype('timedelta64[s]').astype(int) // (24 * 3600)
    day_start = year_start + np.timedelta64(days_since_start, 'D')
    hour_f = (target_time - day_start) / np.timedelta64(1, 'h')
    shift_mit = hour_f * 360 / 24 + 180  # 计算经度偏移
    dim0 = "lon"
    shift_mit_int = int(np.round(shift_mit)) % xr_mit_sel.sizes[dim0]

    # 对经度进行滚动调整

    xr_mit_slon = xr_mit_sel.roll({dim0: shift_mit_int}, roll_coords=True).rename({'lon': 'slon'})
    # 将经度从0~360转换为-180~180
    xr_mit_slon['slon'] = xr_mit_slon['slon'].where(
        xr_mit_slon['slon'] <= 180,
        xr_mit_slon['slon'] - 360
    )
    # 确保经度按顺序排列（避免滚动后坐标顺序混乱）
    xr_mit_slon = xr_mit_slon.sortby('slon')

    # 生成输出文件名
    str_time = pd.to_datetime(target_time).strftime("%Y-%m-%d %H-%M-%S")
    out_fig = os.path.join(output_dir, f"mit_tec {str_time}.png")

    # A4纸宽度21cm，标准边距2.54cm，可插入图片最大宽度约15.6cm
    # 转换为英寸: 15.6cm / 2.54 ≈ 6.14 inches
    a4_width_inch = 6.14
    # 全球地图保持合适的宽高比，纬度范围-60~60，经度-180~180
    # 宽度:高度 = 360:120 = 3:1，但考虑到colorbar，适当增加高度
    fig_height = 2.8  # 约6.35cm，保持扁平

    # 绘制全球TEC图
    fig, ax = plt.subplots(
        1, 1,
        figsize=(a4_width_inch, fig_height),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    add_map_features(ax)

    # 绘制数据，设置colorbar
    im = xr_mit_slon.plot(
        cmap="viridis",
        vmin=0,
        vmax=100,
        ax=ax,
        add_colorbar=True,
        add_labels=False,
        cbar_kwargs={
            'shrink': 0.8,  # colorbar高度
            'aspect': 15,  # colorbar宽高比
            'pad': 0.03,  # colorbar与图的距离
            'label': 'TEC (TECU)',  # colorbar标签
        }
    )

    # 设置colorbar标签字号为10pt
    cbar = im.colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('TEC (TECU)', fontsize=8)

    # 设置坐标轴标签和刻度字号
    ax.set_xlabel('Longitude (°)', fontsize=8)
    ax.set_ylabel('Latitude (°)', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    # 设置刻度范围
    ax.set_xlim(-180, 180)
    ax.set_ylim(-85, 85)

    # 设置刻度间隔
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])

    # 删除标题（科研图通常不需要在图上放标题，放在正文caption中）
    # plt.title(f"MIT Global TEC at {str_time}")

    # 调整布局，确保所有元素可见
    plt.tight_layout()
    plt.savefig(out_fig, dpi=500, bbox_inches='tight')
    plt.close()

    print(f"MIT TEC figure saved to: {out_fig}")


# 使用示例
if __name__ == "__main__":
    # 数据目录（根据实际情况修改）
    data_root = "E:/research"
    # 目标年份和时刻（根据需要修改）
    target_year = 2023
    target_datetime = np.datetime64("2023-01-13T01:10:00")

    # 调用函数绘制TEC图
    plot_mit_global_tec(data_root, target_year, target_datetime)