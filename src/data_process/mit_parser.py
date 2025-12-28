import numpy as np
import pandas as pd
from ppgnss import gnss_utils  # 依赖原项目环境中的库
import xarray as xr


def load_mit_tec(target_time, lon_range, lat_range, mit_file_path, time_tolerance=None):
    """
    加载并提取MIT数据中目标时刻、目标经纬度范围的TEC数据
    """
    # 【恢复日志】打印正在加载的文件
    # print(f"正在加载MIT数据：{mit_file_path}") # 如果觉得太刷屏可以注释这一行，保留下面那行最重要的

    try:
        xr_mit = gnss_utils.loadobject(mit_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"MIT数据文件不存在：{mit_file_path}，请检查路径")
    except Exception as e:
        raise RuntimeError(f"加载MIT数据失败：{str(e)}")

    # 1. 目标时间 (含偏移)
    mit_time_target = target_time + np.timedelta64(150, "s")

    # 2. 寻找最近邻
    xr_mit_time = xr_mit.sel(time=mit_time_target, method="nearest")
    actual_time = pd.to_datetime(xr_mit_time.time.values)

    # 3. 检查时间差
    time_diff = abs((actual_time - pd.to_datetime(mit_time_target)).total_seconds() / 60)

    # 【恢复日志】打印匹配到的具体时间，让您知道它选了哪一帧
    print(f"MIT数据实际匹配时刻：{actual_time} (目标: {mit_time_target}, 偏差: {time_diff:.2f} min)")

    if time_tolerance is not None:
        tol_minutes = time_tolerance.total_seconds() / 60
        if time_diff > tol_minutes:
            raise ValueError(f"时间匹配失败！目标: {mit_time_target}, 最近: {actual_time}, "
                             f"差异: {time_diff:.2f}min > 阈值 {tol_minutes}min")

    # 4. 计算经度旋转 (Longitudinal Shift Logic)
    time_start = xr_mit.time[0]
    hour_f = (actual_time - pd.to_datetime(time_start.values)) / np.timedelta64(1, "h")
    shift_mit = hour_f * 360 / 24 + 180
    shift_mit_int = int(np.round(shift_mit)) % xr_mit_time.sizes["lon"]

    # 5. 执行滚动并重命名
    xr_mit_roll = xr_mit_time.roll(lon=shift_mit_int, roll_coords=True).rename({"lon": "slon"})

    # 6. 经度范围修正 (-180, 180)
    xr_mit_roll['slon'] = xr_mit_roll['slon'].where(
        xr_mit_roll['slon'] <= 180,
        xr_mit_roll['slon'] - 360
    )
    xr_mit_roll = xr_mit_roll.sortby('slon')

    # 7. 区域裁剪
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    xr_mit_tec = xr_mit_roll.sel(
        slon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max)
    )

    if xr_mit_tec.isnull().all():
        raise ValueError(f"MIT TEC数据在范围{lon_range}x{lat_range}内全为空")

    return xr_mit_tec