import numpy as np
import pyIGRF14 as IGRF  # 依赖原项目环境中的库


def compute_igrf_features(lon_grid, lat_grid, year, alt=500):
    """
    计算IGRF地磁特征 (D, I, X, Y, Z)
    参数:
        lon_grid: 经度网格 (2D numpy array)
        lat_grid: 纬度网格 (2D numpy array)
        year: 年份 (int or float)
        alt: 高度 (km), 默认500km
    """
    mag_D = np.zeros(lon_grid.shape)
    mag_I = np.zeros(lon_grid.shape)
    mag_X = np.zeros(lon_grid.shape)
    mag_Y = np.zeros(lon_grid.shape)
    mag_Z = np.zeros(lon_grid.shape)

    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            # 严格保留原始调用的参数逻辑
            d, i_val, _, x, y, z, _ = IGRF.igrf_value(
                lat=lat_grid[i, j],
                lon=lon_grid[i, j],
                alt=alt,
                year=year
            )
            mag_D[i, j] = d
            mag_I[i, j] = i_val
            mag_X[i, j] = x
            mag_Y[i, j] = y
            mag_Z[i, j] = z

    return mag_D, mag_I, mag_X, mag_Y, mag_Z


def compute_day_night_flag(lon_grid, timestamp):
    """计算昼夜标志 (Local Hour based)"""
    hour = timestamp.hour + timestamp.minute / 60
    local_hour = (hour + lon_grid / 15) % 24  # 地方时计算
    day_night = ((local_hour >= 6) & (local_hour < 18)).astype(int)
    return day_night