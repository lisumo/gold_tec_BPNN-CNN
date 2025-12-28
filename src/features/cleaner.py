import numpy as np
from scipy.interpolate import griddata

def bilinear_interpolation(data):
    """对单通道数据进行双线性插值填充缺失值"""
    # 创建网格坐标
    h, w = data.shape
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    # 获取非缺失值的坐标和值
    mask = ~np.isnan(data)
    if np.sum(mask) < 4:  # 至少需要4个点才能进行可靠的双线性插值
        return None  # 无法进行有效插值

    points = np.column_stack((xx[mask].ravel(), yy[mask].ravel()))
    values = data[mask].ravel()

    # 创建目标网格
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # 执行双线性插值
    interpolated = griddata(points, values, grid_points, method='linear', fill_value=np.nan)
    interpolated = interpolated.reshape(h, w)

    # 检查是否还有剩余的NaN
    if np.isnan(interpolated).any():
        # 对剩余NaN使用均值填充
        mean_val = np.nanmean(interpolated)
        interpolated = np.nan_to_num(interpolated, nan=mean_val)

    return interpolated

def handle_missing_values(features, target, target_mask, auxiliary_targets, max_missing_ratio=0.45,
                          use_interpolation=True):
    """处理空间特征和目标的缺失值，根据开关决定是否使用双线性插值"""
    if features.shape[1:] != target.shape:
        print(f"警告：特征与目标形状不匹配 {features.shape[1:]} vs {target.shape}，已跳过")
        return None, None, None, None

    # 如果不使用插值，直接返回原始数据（包含NaN）
    if not use_interpolation:
        return features, target, target_mask, auxiliary_targets

    # 使用插值时的处理逻辑
    total_missing = np.isnan(features).mean() + np.isnan(target).mean()
    if total_missing > max_missing_ratio:
        print(f"样本因缺失值比例过高被过滤（总比例: {total_missing:.2%}）")
        return None, None, None, None

    # 对每个通道进行双线性插值
    features_clean = np.zeros_like(features)
    valid = True

    for c in range(features.shape[0]):
        channel_data = features[c]
        if np.isnan(channel_data).any():
            interpolated = bilinear_interpolation(channel_data)
            if interpolated is None:
                print(f"通道 {c} 无法进行有效插值，样本被过滤")
                valid = False
                break
            features_clean[c] = interpolated
        else:
            features_clean[c] = channel_data

    if not valid:
        return None, None, None, None

    # 处理目标变量的缺失值
    if np.isnan(target).any():
        target_clean = bilinear_interpolation(target)
        if target_clean is None:
            print("目标变量无法进行有效插值，样本被过滤")
            return None, None, None, None
    else:
        target_clean = target

    # 处理辅助目标的缺失值
    auxiliary_clean = np.zeros_like(auxiliary_targets)
    for c in range(auxiliary_targets.shape[0]):
        channel_data = auxiliary_targets[c]
        if np.isnan(channel_data).any():
            interpolated = bilinear_interpolation(channel_data)
            if interpolated is None:
                print(f"辅助目标通道 {c} 无法进行有效插值，样本被过滤")
                return None, None, None, None
            auxiliary_clean[c] = interpolated
        else:
            auxiliary_clean[c] = channel_data

    return features_clean, target_clean, target_mask, auxiliary_clean