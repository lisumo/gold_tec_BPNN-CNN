# src/utils/visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature

# 设置中文显示 (保留原代码逻辑)
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# Color Maps (来自 Script 1)
# ==========================================
def create_gold_cmap():
    """创建GOLD数据专用颜色映射"""
    cmap = LinearSegmentedColormap.from_list(
        'gold_cmap',
        ['darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    )
    cmap.set_over('maroon')
    cmap.set_under('navy')
    return cmap


def create_mit_cmap():
    """创建MIT数据专用颜色映射"""
    cmap = LinearSegmentedColormap.from_list(
        'mit_cmap',
        ['darkblue', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    )
    cmap.set_over('maroon')
    cmap.set_under('navy')
    return cmap


# ==========================================
# Map Features (来自 Script 1)
# ==========================================
def add_map_features(ax):
    """添加地图底图要素（海岸线、国界）"""
    coastline = NaturalEarthFeature(
        category='physical', name='coastline', scale='10m',
        edgecolor='black', facecolor='none'
    )
    ax.add_feature(coastline, linewidth=0.8)
    borders = NaturalEarthFeature(
        category='cultural', name='admin_0_countries', scale='10m',
        edgecolor='black', facecolor='none'
    )
    ax.add_feature(borders, linewidth=0.5)
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)


# ==========================================
# Data Processing Plots (来自 Script 1)
# ==========================================
def plot_mit_gold_comparison(xr_gold_rad, xr_mit_tec, target_time, lon_range, lat_range, output_dir):
    """绘制MIT TEC与GOLD辐射值对比图"""
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, figsize=(20, 10),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    gold_cmap = create_gold_cmap()
    mit_cmap = create_mit_cmap()

    # 绘制GOLD辐射值
    lon_edges = np.linspace(
        xr_gold_rad.longitude.min().values,
        xr_gold_rad.longitude.max().values,
        len(xr_gold_rad.longitude) + 1
    )
    lat_edges = np.linspace(
        xr_gold_rad.latitude.min().values,
        xr_gold_rad.latitude.max().values,
        len(xr_gold_rad.latitude) + 1
    )
    mesh1 = ax1.pcolormesh(
        lon_edges, lat_edges, xr_gold_rad.values,
        cmap=gold_cmap, vmin=0, vmax=1000,
        shading='flat', transform=ccrs.PlateCarree()
    )
    add_map_features(ax1)
    ax1.set_title(
        f'GOLD 135.6nm Radiance\nTarget Time: {target_time.strftime("%Y-%m-%d %H:%M")}',
        fontsize=16, pad=20
    )
    cbar1 = fig.colorbar(mesh1, ax=ax1, label='Radiance (Rayleighs/nm)', pad=0.02, aspect=30, extend='both')
    cbar1.set_label('Radiance (Rayleighs/nm)', fontsize=12)

    # 绘制MIT TEC
    slon_edges = np.linspace(
        xr_mit_tec.slon.min().values,
        xr_mit_tec.slon.max().values,
        len(xr_mit_tec.slon) + 1
    )
    lat_edges_mit = np.linspace(
        xr_mit_tec.lat.min().values,
        xr_mit_tec.lat.max().values,
        len(xr_mit_tec.lat) + 1
    )
    mesh2 = ax2.pcolormesh(
        slon_edges, lat_edges_mit, xr_mit_tec.values,
        cmap=mit_cmap, vmin=0, vmax=150,
        shading='flat', transform=ccrs.PlateCarree()
    )
    add_map_features(ax2)
    ax2.set_title(
        f'MIT TEC\nMatched Time: {pd.to_datetime(xr_mit_tec.time.values).strftime("%Y-%m-%d %H:%M")}',
        fontsize=16, pad=20
    )
    cbar2 = fig.colorbar(mesh2, ax=ax2, label='TEC (TECU)', pad=0.02, aspect=30, extend='both')
    cbar2.set_label('TEC (TECU)', fontsize=12)

    plt.tight_layout()
    fig_name = f"MIT_GOLD_Compare_{target_time.strftime('%Y%m%d_%H%M')}_Lon{lon_range[0]}_{lon_range[1]}_Lat{lat_range[0]}_{lat_range[1]}.png"
    fig_path = os.path.join(output_dir, fig_name)
    plt.savefig(fig_path, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存：{fig_path}")
    return fig_path


# ==========================================
# Training & Evaluation Plots (来自 Script 3)
# ==========================================
def plot_training_validation_loss(history):
    """绘制训练与验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='训练损失', linewidth=1.5)
    plt.plot(history['val_loss'], label='验证损失', linewidth=1.5)
    plt.title('训练与验证损失', fontsize=12, fontweight='bold')
    plt.xlabel('轮次', fontsize=10)
    plt.ylabel('损失', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_prediction_vs_truth(y_test_valid, y_pred_valid):
    """绘制预测值 vs 真实值散点图"""
    plt.figure(figsize=(10, 6))
    if len(y_test_valid) > 0:
        plt.scatter(y_test_valid, y_pred_valid, alpha=0.2, s=10)
        plt.plot([y_test_valid.min(), y_test_valid.max()],
                 [y_test_valid.min(), y_pred_valid.max()],
                 'r--', label='理想预测线', linewidth=2)
        plt.title('预测值 vs 真实值', fontsize=12, fontweight='bold')
        plt.xlabel('真实TEC值', fontsize=10)
        plt.ylabel('预测TEC值', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("警告：没有有效数据点可用于绘制预测值 vs 真实值图")


def plot_residual_histogram(residuals):
    """绘制残差分布直方图"""
    plt.figure(figsize=(10, 6))
    if len(residuals) > 0:
        n, bins, patches = plt.hist(residuals, bins=20, alpha=0.7, label='残差分布', color='#1f77b4')
        plt.title('残差分布直方图', fontsize=12, fontweight='bold')
        plt.xlabel('残差值 (预测值-真实值)', fontsize=10)
        plt.ylabel('频数', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    else:
        print("警告：没有有效残差数据可用于绘制直方图")


# ==========================================
# Single Sample Visualization (来自 Script 3)
# ==========================================
def plot_sample_heatmap(data, mask, title, vmin=None, vmax=None, cmap='viridis'):
    """通用的样本热力图绘制辅助函数 (重构提取的公共逻辑)"""
    plt.figure(figsize=(8, 6))
    # 应用掩码
    data_masked = np.where(mask, data, np.nan)
    im = plt.imshow(data_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()
    plt.colorbar(im, label='Value', shrink=0.8)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('经度索引', fontsize=10)
    plt.ylabel('纬度索引', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_true_tec_distribution(y_test_original, mit_masks_test, sample_idx):
    true_tec = y_test_original[sample_idx, 0]
    mask = mit_masks_test[sample_idx, 0]
    plot_sample_heatmap(true_tec, mask, f'样本 {sample_idx} 的真实TEC分布', cmap='viridis')


def plot_cnn_prediction(cnn_output_original, gold_rad_masks_test, sample_idx, vmin, vmax):
    cnn_tec = cnn_output_original[sample_idx, 0]
    mask = gold_rad_masks_test[sample_idx, 0]
    plot_sample_heatmap(cnn_tec, mask, f'样本 {sample_idx} 的CNN预测', vmin=vmin, vmax=vmax)


def plot_bpnn_background(bpnn_output_original, gold_rad_masks_test, sample_idx, vmin, vmax):
    bpnn_tec = bpnn_output_original[sample_idx, 0]
    mask = gold_rad_masks_test[sample_idx, 0]
    plot_sample_heatmap(bpnn_tec, mask, f'样本 {sample_idx} 的BPNN背景场', vmin=vmin, vmax=vmax)


def plot_fusion_result(y_pred_original, gold_rad_masks_test, sample_idx, vmin, vmax):
    fused_tec = y_pred_original[sample_idx, 0]
    mask = gold_rad_masks_test[sample_idx, 0]
    plot_sample_heatmap(fused_tec, mask, f'样本 {sample_idx} 的融合结果', vmin=vmin, vmax=vmax)


def plot_residual_map(pred, true_val, mask, title):
    """绘制残差图的辅助函数"""
    plt.figure(figsize=(8, 6))
    residual = pred - true_val
    residual_masked = np.where(mask, residual, np.nan)

    # 动态计算范围
    residual_max = max(np.nanmax(np.abs(residual_masked)), 1e-6)

    im = plt.imshow(residual_masked, cmap='RdBu_r', vmin=-residual_max, vmax=residual_max)
    plt.gca().invert_yaxis()
    plt.colorbar(im, label='残差值', shrink=0.8)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('经度索引', fontsize=10)
    plt.ylabel('纬度索引', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_cnn_residual(cnn_output_original, y_test_original, mit_masks_test, gold_rad_masks_test, sample_idx):
    cnn_tec = cnn_output_original[sample_idx, 0]
    true_tec = y_test_original[sample_idx, 0]
    intersection_mask = mit_masks_test[sample_idx, 0] & gold_rad_masks_test[sample_idx, 0]
    plot_residual_map(cnn_tec, true_tec, intersection_mask, f'样本 {sample_idx} 的CNN残差')


def plot_fusion_residual(y_pred_original, y_test_original, mit_masks_test, gold_rad_masks_test, sample_idx):
    fused_tec = y_pred_original[sample_idx, 0]
    true_tec = y_test_original[sample_idx, 0]
    intersection_mask = mit_masks_test[sample_idx, 0] & gold_rad_masks_test[sample_idx, 0]
    plot_residual_map(fused_tec, true_tec, intersection_mask, f'样本 {sample_idx} 的融合残差')