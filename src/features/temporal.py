import numpy as np


def encode_time_features(timestamp):
    """
    生成全局时间特征 (4维)
    1. Hour sin/cos
    2. Julian Day sin/cos
    """
    hour = timestamp.hour + timestamp.minute / 60
    time_angle = 2 * np.pi * hour / 24  # 小时周期（0-24h）

    julian_day = timestamp.timetuple().tm_yday
    julian_angle = 2 * np.pi * julian_day / 365  # 年周期（0-365天）

    time_features = np.array([
        np.sin(time_angle),  # 小时正弦特征
        np.cos(time_angle),  # 小时余弦特征
        np.sin(julian_angle),  # 年积日正弦特征
        np.cos(julian_angle)  # 年积日余弦特征
    ])

    return time_features