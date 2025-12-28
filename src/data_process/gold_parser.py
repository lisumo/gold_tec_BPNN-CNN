import os
import numpy as np
import pandas as pd
import netCDF4 as nc

def parse_gold_time_from_path(file_path):
    """从GOLD文件路径解析时间"""
    file_name = os.path.basename(file_path)
    try:
        # 逻辑来源：原始脚本 gold_mit_data_processor.py
        # 兼容两种文件命名格式的处理逻辑
        if "GOLD_NI1_gridded_data_" in file_name:
            # 处理中间生成的NC文件格式: GOLD_NI1_gridded_data_2023_292_00_10.nc
            prefix = 'GOLD_NI1_gridded_data_'
            date_str = file_name.split(prefix)[1].split('.nc')[0]
            year, julian_day, hour, minute = date_str.split('_')
            base_date = pd.Timestamp(f"{year}-01-01")
            target_date = base_date + pd.Timedelta(days=int(julian_day) - 1)
            return pd.Timestamp(f"{target_date.date()} {int(hour):02d}:{int(minute):02d}:00")
        else:
            # 处理原始L1C文件格式
            parts = file_name.split('_')
            year = int(parts[-4])
            julian_day = int(parts[-3])
            hour = int(parts[-2])
            minute = int(parts[-1].split('.')[0])
            base_date = pd.Timestamp(f"{year}-01-01")
            target_date = base_date + pd.Timedelta(days=julian_day - 1)
            return pd.Timestamp(f"{target_date.date()} {hour:02d}:{minute:02d}:00")
    except (IndexError, ValueError) as e:
        raise ValueError(f"无法从GOLD文件路径解析时间：{file_path}，错误：{str(e)}")

def process_ni1_file(file_path, lon_min, lon_max, lat_min, lat_max):
    """处理单个NI1文件，返回指定区域的经纬度、辐射值和匹配的波长"""
    nc_file = nc.Dataset(file_path)
    lat = nc_file.variables['REFERENCE_POINT_LAT'][:]
    lon = nc_file.variables['REFERENCE_POINT_LON'][:]
    wavelength = nc_file.variables['WAVELENGTH'][:]
    radiance = nc_file.variables['RADIANCE'][:]

    # 提取波长逻辑
    if wavelength.ndim == 3:
        wavelength = wavelength[10, 10, :]
    target_wave = 135.6
    wave_index = np.argmin(np.abs(wavelength - target_wave))
    closest_wave = wavelength[wave_index]
    print(f"NI1文件 {file_path} 匹配到最接近的波长：{closest_wave:.5f}nm")

    # 提取辐射值
    radiance = radiance[..., wave_index]
    data = {
        '经度': lon.flatten(),
        '纬度': lat.flatten(),
        '辐射值': radiance.flatten()
    }
    df = pd.DataFrame(data).dropna()

    # 区域初步筛选
    region_mask = (df['经度'] >= lon_min) & (df['经度'] <= lon_max) & \
                  (df['纬度'] >= lat_min) & (df['纬度'] <= lat_max)
    lon_region = df['经度'][region_mask]
    lat_region = df['纬度'][region_mask]
    radiance_region = df['辐射值'][region_mask]
    nc_file.close()

    region_df = pd.DataFrame({
        '经度': lon_region,
        '纬度': lat_region,
        '辐射值': radiance_region,
    })
    return region_df, closest_wave