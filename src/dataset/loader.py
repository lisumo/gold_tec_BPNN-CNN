import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
from torch.utils.data import Dataset
from ..data_process.gold_parser import parse_gold_time_from_path
from ..data_process.gridder import align_grids
from ..features.physics import compute_igrf_features, compute_day_night_flag
from ..features.temporal import encode_time_features
from ..features.cleaner import handle_missing_values
from config import Config


def match_files_by_time(gold_dirs, mit_dir, time_tolerance_minutes=10):
    """
    匹配时间相近的GOLD和MIT文件对
    支持 gold_dirs 为单个路径字符串或路径列表
    """
    # 兼容性处理：如果是字符串，转为列表
    if isinstance(gold_dirs, str):
        gold_dirs = [gold_dirs]

    if not os.path.exists(mit_dir):
        raise FileNotFoundError(f"MIT目录不存在: {mit_dir}")

    # 1. 收集所有 GOLD 文件 (遍历所有年份目录)
    all_gold_files = []  # 存储 (filename, full_path)
    print(f"开始扫描 GOLD 数据目录...")

    for d in gold_dirs:
        if os.path.exists(d):
            # 注意：这里假设您使用的是预处理后的 gridded 数据
            # 如果您直接使用 level1c 数据，请修改这里的文件名前缀匹配
            files = [f for f in os.listdir(d) if f.startswith('GOLD_NI1_gridded_data_') and f.endswith('.nc')]
            print(f" -> 在目录 {d} 中找到 {len(files)} 个文件")
            for f in files:
                all_gold_files.append((f, os.path.join(d, f)))
        else:
            print(f" [警告] GOLD目录不存在: {d}")

    # 2. 收集 MIT 文件
    mit_files = [f for f in os.listdir(mit_dir) if f.startswith('MIT_TEC_data_') and f.endswith('.nc')]
    print(f"在 MIT 目录找到 {len(mit_files)} 个文件")

    if not all_gold_files or not mit_files:
        raise ValueError("GOLD 或 MIT 目录中未找到符合条件的文件 (请检查文件名前缀)")

    time_tolerance = timedelta(minutes=time_tolerance_minutes)
    file_pairs = []

    # 3. 进行时间匹配
    # 为了提高效率，这里依然采用双重循环，但在数据量极大时可考虑哈希索引
    print("正在匹配文件时间戳...")
    matched_count = 0

    for g_file, g_path in all_gold_files:
        try:
            # 解析 GOLD 时间
            g_time = parse_gold_time_from_path(g_path)
        except Exception as e:
            # print(f"无法解析文件时间: {g_file}")
            continue

        for m_file in mit_files:
            try:
                # 解析 MIT 时间: MIT_TEC_data_2023_200_1200.nc
                m_date_str = m_file.split('MIT_TEC_data_')[1].split('.nc')[0]
                year, julian_day, hour, minute = m_date_str.split('_')
                m_base_date = pd.Timestamp(f"{year}-01-01")
                m_time = m_base_date + timedelta(days=int(julian_day) - 1, hours=int(hour), minutes=int(minute))

                if abs(g_time - m_time) <= time_tolerance:
                    file_pairs.append((g_path, os.path.join(mit_dir, m_file), g_time))
                    matched_count += 1
                    break  # 找到一个匹配就跳出 MIT 循环
            except Exception:
                continue

    print(f"最终匹配到 {len(file_pairs)} 对文件")
    return file_pairs


def assemble_features(gold_ds, mit_ds, timestamp):
    """组装空间和时间特征 (保持不变)"""
    lon = gold_ds.longitude.values
    lat = gold_ds.latitude.values
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    lon_rad = np.radians(lon_grid)
    lat_rad = np.radians(lat_grid)

    lon_sin = np.sin(lon_rad)
    lon_cos = np.cos(lon_rad)
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)

    mag_D, mag_I, mag_X, mag_Y, mag_Z = compute_igrf_features(lon_grid, lat_grid, timestamp.year)
    day_night = compute_day_night_flag(lon_grid, timestamp)

    radiance = gold_ds.radiance.values
    radiance_log = np.log1p(radiance)
    gold_radiance_mask = ~np.isnan(radiance)

    spatial_features = np.stack([
        radiance, radiance_log, lon_sin, lon_cos, lat_sin, lat_cos,
        mag_D, mag_I, mag_X, mag_Y, mag_Z, day_night
    ], axis=0)

    time_features = encode_time_features(timestamp)

    tec_data = mit_ds.tec.values.squeeze()
    tec_mask = ~np.isnan(tec_data)

    auxiliary_targets = np.stack([radiance, radiance_log], axis=0)

    return spatial_features, time_features, tec_data, tec_mask, auxiliary_targets, gold_radiance_mask


class GoldMit2DDataset(Dataset):
    def __init__(self, gold_dir, mit_dir, time_tolerance_minutes=10, use_interpolation=True):
        self.use_interpolation = use_interpolation

        # gold_dir 现在可以是列表，由 match_files_by_time 内部处理
        self.file_pairs = match_files_by_time(gold_dir, mit_dir, time_tolerance_minutes) if gold_dir and mit_dir else []

        # 存储时间戳用于周期划分
        self.timestamps = []
        self.samples = self._load_samples()

        if len(self.samples) == 0:
            print("警告: 数据集为空！")

    def _rename_mit_tec_variable(self, mit_ds):
        if 'tec' in mit_ds.data_vars:
            return mit_ds
        required_dims_set = [{'lat', 'slon', 'time'}, {'lat', 'lon', 'time'}]
        tec_var_name = None
        for var_name in mit_ds.data_vars:
            var_dims = set(mit_ds[var_name].dims)
            if any(req_dims.issubset(var_dims) for req_dims in required_dims_set):
                tec_var_name = var_name
                break
        if tec_var_name:
            mit_ds = mit_ds.rename({tec_var_name: 'tec'})
        else:
            tec_var_name = next(iter(mit_ds.data_vars))
            mit_ds = mit_ds.rename({tec_var_name: 'tec'})
        return mit_ds

    def _load_samples(self):
        samples = []
        self.timestamps = []  # 重置时间戳列表

        for idx, (gold_path, mit_path, timestamp) in enumerate(self.file_pairs):
            try:
                if (idx + 1) % 50 == 0:
                    print(f"处理进度: {idx + 1}/{len(self.file_pairs)}")

                gold_ds = xr.open_dataset(gold_path)
                mit_ds = xr.open_dataset(mit_path)
                mit_ds = self._rename_mit_tec_variable(mit_ds)

                gold_aligned, mit_aligned = align_grids(gold_ds, mit_ds)

                spatial_features, time_features, target, target_mask, auxiliary_targets, gold_radiance_mask = assemble_features(
                    gold_aligned, mit_aligned, timestamp
                )

                spatial_clean, target_clean, target_mask_clean, auxiliary_clean = handle_missing_values(
                    spatial_features, target, target_mask, auxiliary_targets,
                    max_missing_ratio=0.45,
                    use_interpolation=self.use_interpolation
                )

                if spatial_clean is not None:
                    # 格式化时间字符串
                    time_str = str(timestamp)

                    samples.append((
                        torch.tensor(spatial_clean, dtype=torch.float32),
                        torch.tensor(time_features, dtype=torch.float32),
                        torch.tensor(target_clean, dtype=torch.float32),
                        torch.tensor(target_mask_clean, dtype=torch.bool),
                        torch.tensor(auxiliary_clean, dtype=torch.float32),
                        torch.tensor(gold_radiance_mask, dtype=torch.bool),
                        time_str
                    ))
                    # 记录有效样本的时间戳
                    self.timestamps.append(timestamp)

                gold_ds.close()
                mit_ds.close()

            except Exception as e:
                # 可以选择打印错误或静默跳过
                continue

        print(f"\n预处理完成，共加载 {len(samples)} 个有效样本")
        return samples

    def save_dataset(self, save_path):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.samples, save_path)
        print(f"数据集已保存至: {save_path}")

    @staticmethod
    def load_dataset(load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"数据集文件不存在: {load_path}")
        samples = torch.load(load_path)
        print(f"已加载数据集，包含 {len(samples)} 个样本")
        dataset = GoldMit2DDataset(gold_dir=None, mit_dir=None, use_interpolation=False)
        dataset.samples = samples
        return dataset

    # 【关键修改】周期性划分方法：支持跨年
    def split_by_period(self, start_doy, window_size=50, train_days=35, val_days=10):
        """
        根据绝对日期进行周期性划分，完美支持跨年 (2023->2024)
        """
        train_samples = []
        val_samples = []
        test_samples = []

        if not self.timestamps:
            print("警告: 无时间戳信息，无法进行周期划分，返回空数据集")
            return self.from_samples([]), self.from_samples([]), self.from_samples([])

        # 1. 找到数据集中的最早时间，作为 Day 1
        # 这样无论数据从哪天开始，都是连续计算的
        min_date = min(self.timestamps).date()
        print(f"\n📊 执行周期性划分 (Window={window_size})")
        print(f"   基准起始日期 (Day 1): {min_date}")
        print(
            f"   划分方案: Day 1-{train_days} (Train) | Day {train_days + 1}-{train_days + val_days} (Val) | Day {train_days + val_days + 1}-{window_size} (Test)")

        for i, sample in enumerate(self.samples):
            ts = self.timestamps[i]

            # 2. 计算距离基准日期的天数 (从0开始)
            # 2023-12-31 是 Day X，2024-01-01 就是 Day X+1，保证了连续性
            day_diff = (ts.date() - min_date).days

            # 3. 计算在当前周期内的位置 (1 到 window_size)
            cycle_day = (day_diff % window_size) + 1

            if cycle_day <= train_days:
                train_samples.append(sample)
            elif cycle_day <= (train_days + val_days):
                val_samples.append(sample)
            else:
                test_samples.append(sample)

        print(f"📈 划分结果: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

        return (
            self.from_samples(train_samples),
            self.from_samples(val_samples),
            self.from_samples(test_samples)
        )

    @classmethod
    def from_samples(cls, samples):
        """辅助方法: 从样本列表创建 Dataset"""
        instance = cls(None, None, use_interpolation=False)
        instance.samples = samples
        instance.file_pairs = []
        return instance

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]