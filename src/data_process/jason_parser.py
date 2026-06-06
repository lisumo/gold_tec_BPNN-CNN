import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta


def parse_jason_filename_time(filename):
    """
    从 Jason-3 文件名中解析开始和结束时间
    格式示例: JA3_GPN_2PfP000_117_20160212_011109_20160212_020721.nc
    """
    try:
        parts = filename.split('_')
        # parts[4] 是开始日期 (20160212), parts[5] 是开始时间 (011109)
        # parts[6] 是结束日期 (20160212), parts[7] 是结束时间 (020721)
        start_str = parts[4] + parts[5]
        end_str = parts[6] + parts[7].replace('.nc', '')

        start_time = pd.to_datetime(start_str, format='%Y%m%d%H%M%S')
        end_time = pd.to_datetime(end_str, format='%Y%m%d%H%M%S')
        return start_time, end_time
    except Exception as e:
        # print(f"文件名解析失败: {filename}")
        return None, None


def find_jason_files_for_timestamp(jason_dir, target_time, tolerance_minutes=10):
    """
    在指定目录下查找覆盖目标时间点的 Jason 文件
    """
    if not os.path.exists(jason_dir):
        return []

    # 获取目录下所有 nc 文件
    all_files = glob.glob(os.path.join(jason_dir, "JA3_GPN_*.nc"))
    matched_files = []

    target_ts = pd.to_datetime(target_time)
    tol = timedelta(minutes=tolerance_minutes)

    for f_path in all_files:
        f_name = os.path.basename(f_path)
        start_t, end_t = parse_jason_filename_time(f_name)

        if start_t is None: continue

        # 检查时间区间是否有交集 (文件时间范围 vs 目标时间点 +/- 容忍度)
        # 只要文件的时间范围与 [target-tol, target+tol] 有重叠即可
        overlap_start = max(start_t, target_ts - tol)
        overlap_end = min(end_t, target_ts + tol)

        if overlap_start < overlap_end:
            matched_files.append(f_path)

    return matched_files


def read_and_filter_jason_data(file_paths, target_time, lon_range, lat_range, time_tol_minutes=15):
    """
    读取多个 Jason 文件，并筛选出符合时空条件且位于海洋的点
    【新增清洗逻辑】：
    1. rad_rain_flag == 0 (剔除受降雨影响的点)
    2. iono_cor_alt_qual == 0 (剔除电离层修正质量差的点)
    """
    all_points = []
    target_ts = pd.to_datetime(target_time)

    for f_path in file_paths:
        try:
            # 读取基础组和 Ku 组
            ds_base = xr.open_dataset(f_path, group='data_01')
            ds_ku = xr.open_dataset(f_path, group='data_01/ku')

            # 提取基础变量
            time = pd.to_datetime(ds_base['time'].values)
            lat = ds_base['latitude'].values
            lon = ds_base['longitude'].values
            # 经度转换 0-360 -> -180~180
            lon = np.where(lon > 180, lon - 360, lon)

            # 地表类型 (0: Open Ocean)
            if 'rad_surface_type_flag' in ds_base:
                surf = ds_base['rad_surface_type_flag'].values
            else:
                surf = ds_base['surface_type'].values

            # 【关键修改】读取质量标志位
            # 1. 雨标志 (rad_rain_flag) - 通常在 data_01
            if 'rad_rain_flag' in ds_base:
                rain_flag = ds_base['rad_rain_flag'].values
            else:
                # 假如文件里真没有这个变量，保守起见设为0（不剔除）或打印警告
                rain_flag = np.zeros(len(time))

                # 2. 电离层质量标志 (iono_cor_alt_qual) - 可能在 data_01 或 data_01/ku
            if 'iono_cor_alt_qual' in ds_ku:
                iono_qual = ds_ku['iono_cor_alt_qual'].values
            elif 'iono_cor_alt_qual' in ds_base:
                iono_qual = ds_base['iono_cor_alt_qual'].values
            else:
                iono_qual = np.zeros(len(time))

            # 电离层修正量
            if 'iono_cor_alt_ku' in ds_ku:
                iono = ds_ku['iono_cor_alt_ku'].values
            elif 'iono_cor_alt' in ds_ku:
                iono = ds_ku['iono_cor_alt'].values
            else:
                ds_base.close()
                ds_ku.close()
                continue

            # 构建 DataFrame
            df = pd.DataFrame({
                'time': time, 'lat': lat, 'lon': lon,
                'iono_cor': iono, 'surface': surf,
                'rain_flag': rain_flag,
                'iono_qual': iono_qual
            })

            ds_base.close()
            ds_ku.close()

            # --- 筛选逻辑 ---

            # 1. 时间筛选
            time_diff = (df['time'] - target_ts).abs()
            df = df[time_diff <= timedelta(minutes=time_tol_minutes)]

            # 2. 空间筛选
            df = df[
                (df['lon'] >= lon_range[0]) & (df['lon'] <= lon_range[1]) &
                (df['lat'] >= lat_range[0]) & (df['lat'] <= lat_range[1])
                ]

            # 3. 海洋筛选 & 有效值筛选
            df = df[df['surface'] == 0]

            # 4. 【关键修改】质量标志筛选
            # 剔除有雨的点 (rain_flag != 0)
            df = df[df['rain_flag'] == 0]
            # 剔除电离层修正质量差的点 (iono_qual != 0)
            df = df[df['iono_qual'] == 0]

            # 5. 去除空值
            df = df.dropna(subset=['iono_cor'])

            if len(df) > 0:
                # 计算 TEC
                f_ku = 13.575e9
                K = 40.3
                TECU_SCALE = 1e16
                df['jason_tec'] = -1 * (df['iono_cor'] * f_ku ** 2) / (K * TECU_SCALE)

                # 6. 物理范围筛选
                df = df[(df['jason_tec'] > 0) & (df['jason_tec'] < 300)]

                all_points.append(df)

        except Exception as e:
            print(f"读取文件出错 {os.path.basename(f_path)}: {e}")
            continue

    if not all_points:
        return pd.DataFrame()

    return pd.concat(all_points, ignore_index=True)