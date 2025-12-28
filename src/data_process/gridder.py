import numpy as np
import xarray as xr

def grid_data_to_xarray(df, lon_min, lon_max, lat_min, lat_max, grid_spacing=1.0):
    """将数据格网化为指定间隔的栅格数据，并返回xarray.DataArray"""
    num_lon_cells = int((lon_max - lon_min) // grid_spacing)
    num_lat_cells = int((lat_max - lat_min) // grid_spacing)
    actual_lon_max = lon_min + num_lon_cells * grid_spacing
    actual_lat_max = lat_min + num_lat_cells * grid_spacing

    lon_grid = np.arange(lon_min, actual_lon_max + grid_spacing, grid_spacing)
    lat_grid = np.arange(lat_min, actual_lat_max + grid_spacing, grid_spacing)

    # 这里的copy是为了不影响原始df
    df = df.copy()
    df['lon_bin'] = np.floor((df['经度'] - lon_min) / grid_spacing).astype(int)
    df['lat_bin'] = np.floor((df['纬度'] - lat_min) / grid_spacing).astype(int)
    valid_bins = (df['lon_bin'] >= 0) & (df['lon_bin'] < num_lon_cells) & \
                 (df['lat_bin'] >= 0) & (df['lat_bin'] < num_lat_cells)
    df_valid = df[valid_bins]

    # 分组求平均
    grid_avg = df_valid.groupby(['lat_bin', 'lon_bin'])['辐射值'].mean().reset_index()
    grid_data = np.full((len(lat_grid), len(lon_grid)), np.nan)

    for _, row in grid_avg.iterrows():
        lat_idx = int(row['lat_bin'])
        lon_idx = int(row['lon_bin'])
        if lat_idx < len(lat_grid) and lon_idx < len(lon_grid):
            grid_data[lat_idx, lon_idx] = row['辐射值']

    da = xr.DataArray(
        grid_data,
        name='radiance',
        dims=['latitude', 'longitude'],
        coords={
            'latitude': lat_grid,
            'longitude': lon_grid
        },
        attrs={
            'actual_lon_min': lon_min,
            'actual_lon_max': actual_lon_max,
            'actual_lat_min': lat_min,
            'actual_lat_max': actual_lat_max,
            'grid_spacing': grid_spacing
        }
    )
    return da

def mask_gold_region(xr_gold_rad, lon_min, lon_max, lat_min, lat_max):
    """将特定矩形区域内的135.6nm辐射值填充为NaN"""
    masked_gold = xr_gold_rad.copy()
    masked_gold = masked_gold.where(
        ~((masked_gold.longitude >= lon_min) &
          (masked_gold.longitude <= lon_max) &
          (masked_gold.latitude >= lat_min) &
          (masked_gold.latitude <= lat_max))
    )
    return masked_gold

def align_grids(gold_ds, mit_ds):
    """对齐GOLD和MIT数据的经纬度网格 (求交集)"""
    gold_lon = gold_ds.longitude.values
    gold_lat = gold_ds.latitude.values
    # 兼容 slon 或 lon 命名
    mit_lon = mit_ds.slon.values if 'slon' in mit_ds.dims else mit_ds.lon.values
    mit_lat = mit_ds.lat.values

    lon_min = max(gold_lon.min(), mit_lon.min())
    lon_max = min(gold_lon.max(), mit_lon.max())
    lat_min = max(gold_lat.min(), mit_lat.min())
    lat_max = min(gold_lat.max(), mit_lat.max())

    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("经纬度网格无交集")

    gold_lon_mask = (gold_lon >= lon_min) & (gold_lon <= lon_max)
    gold_lat_mask = (gold_lat >= lat_min) & (gold_lat <= lat_max)
    mit_lon_mask = (mit_lon >= lon_min) & (mit_lon <= lon_max)
    mit_lat_mask = (mit_lat >= lat_min) & (mit_lat <= lat_max)

    if not np.any(gold_lon_mask) or not np.any(gold_lat_mask):
        raise ValueError("GOLD数据在交集范围内无有效网格点")
    if not np.any(mit_lon_mask) or not np.any(mit_lat_mask):
        raise ValueError("MIT数据在交集范围内无有效网格点")

    gold_valid_lon_idx = np.where(gold_lon_mask)[0]
    gold_valid_lat_idx = np.where(gold_lat_mask)[0]
    mit_valid_lon_idx = np.where(mit_lon_mask)[0]
    mit_valid_lat_idx = np.where(mit_lat_mask)[0]

    gold_cropped = gold_ds.isel(longitude=gold_valid_lon_idx, latitude=gold_valid_lat_idx)
    mit_cropped = mit_ds.isel(
        slon=mit_valid_lon_idx if 'slon' in mit_ds.dims else {'lon': mit_valid_lon_idx},
        lat=mit_valid_lat_idx
    )

    return gold_cropped, mit_cropped