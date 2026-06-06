import sys
import os
import glob
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import Config
from src.data_process.gold_parser import process_ni1_file, parse_gold_time_from_path
from src.data_process.mit_parser import load_mit_tec
from src.data_process.gridder import grid_data_to_xarray, mask_gold_region


# 【修改】不再导入绘图模块，避免 SimHei 报错
# from src.utils.visualization import plot_mit_gold_comparison

def generate_nc_file_list():
    """
    根据 Config 配置生成文件列表。
    【保留严格匹配逻辑】：
    1. 遍历日期 (支持跨年)
    2. 遍历 Config.TIME_PAIRS
    3. 成对查找 CHA 和 CHB 文件
    """

    valid_files_flat = []

    # 1. 计算起始日期
    start_date = datetime(Config.START_YEAR, 1, 1) + timedelta(days=Config.START_DOY - 1)

    print(f"🔄 开始扫描数据...")
    print(f"📅 扫描范围: {start_date.strftime('%Y-%m-%d')} 起，共 {Config.TOTAL_DAYS} 天")
    print(f"⏰ 匹配逻辑: 严格对应 Config.TIME_PAIRS (CHA + CHB 配对)")

    last_year = -1

    # 2. 基于总天数循环 (解决跨年 DOY 重置问题)
    for i in range(Config.TOTAL_DAYS):
        current_date = start_date + timedelta(days=i)

        current_year = current_date.year
        current_doy = current_date.timetuple().tm_yday
        day_str = f"{current_doy:03d}"

        # 打印跨年信息
        if current_year != last_year:
            print(f"👉 正处理 {current_year} 年数据...")
            last_year = current_year

        # 3. 根据年份动态选择数据源路径
        if Config.HAS_DOY_SUBFOLDERS:
            # 简单的年份判断：2023用第一个路径，2024用第二个
            if current_year == 2023 and len(Config.RAW_GOLD_PATHS) > 0:
                base_path = Config.RAW_GOLD_PATHS[0]
            elif current_year == 2024 and len(Config.RAW_GOLD_PATHS) > 1:
                base_path = Config.RAW_GOLD_PATHS[1]
            else:
                base_path = Config.RAW_GOLD_PATHS[0]  # 默认回退

            day_dir = os.path.join(base_path, day_str)
        else:
            day_dir = Config.RAW_GOLD_PATHS[0]

        if Config.HAS_DOY_SUBFOLDERS and not os.path.exists(day_dir):
            if i % 10 == 0:
                print(f"⚠️ [目录缺失] 找不到文件夹: {day_dir}")
            continue

        # 4. 严格匹配 CHA/CHB
        files_found_today = 0
        search_pattern_debug = ""

        for time_cha, time_chb in Config.TIME_PAIRS:
            # 构造文件名匹配模式 (注意使用 current_year)
            pat_cha = f"*CHA*NI1*_{current_year}_{day_str}_{time_cha}{Config.FIXED_SUFFIX}*.nc"
            pat_chb = f"*CHB*NI1*_{current_year}_{day_str}_{time_chb}{Config.FIXED_SUFFIX}*.nc"

            path_cha_pattern = os.path.join(day_dir, pat_cha)
            path_chb_pattern = os.path.join(day_dir, pat_chb)

            if not search_pattern_debug: search_pattern_debug = path_cha_pattern

            files_cha = glob.glob(path_cha_pattern)
            files_chb = glob.glob(path_chb_pattern)

            if files_cha and files_chb:
                files_cha.sort()
                files_chb.sort()
                valid_files_flat.append(files_cha[0])
                valid_files_flat.append(files_chb[0])
                files_found_today += 1

        if files_found_today == 0 and current_doy == 1:
            print(f"⚠️ [文件缺失] {current_year}年第1天未找到文件。搜索模式示例: {search_pattern_debug}")

    # 保存列表
    with open(Config.NC_LIST_FILENAME, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_files_flat))

    pair_count = len(valid_files_flat) // 2
    print(f"✅ 文件列表生成完成: {Config.NC_LIST_FILENAME}")
    print(f"📊 总共找到 {pair_count} 对文件 (共 {len(valid_files_flat)} 个文件)")

    if pair_count == 0:
        print("❌ 警告：未找到任何文件对！请检查路径或 Config 中的文件后缀设置。")

    return valid_files_flat


def main():
    files = generate_nc_file_list()
    if not files: return

    for dir_path in [Config.OUTPUT_DIR_GOLD, Config.OUTPUT_DIR_MIT]:
        os.makedirs(dir_path, exist_ok=True)

    print("🚀 开始批量预处理 (容忍度模式)...")

    # 计数器
    skipped_count = 0
    processed_count = 0

    for i in range(0, len(files), 2):
        if i + 1 >= len(files): break

        nh_file = files[i]
        sh_file = files[i + 1]

        nh_filename = os.path.basename(nh_file)
        try:
            date_info = nh_filename.split('_')[4:8]
            date_str = '_'.join(date_info)
            print(f"Processing: {date_str} ...", end='\r')
        except:
            print(f"\nSkipping file with unexpected name format: {nh_filename}")
            continue

        try:
            # === GOLD 处理 ===
            nh_df, _ = process_ni1_file(nh_file, Config.MASK_LON_MIN, Config.MASK_LON_MAX, 0, Config.MASK_LAT_MAX)
            sh_df, _ = process_ni1_file(sh_file, Config.MASK_LON_MIN, Config.MASK_LON_MAX, Config.MASK_LAT_MIN, 0)

            all_df = pd.concat([nh_df, sh_df], ignore_index=True)
            if len(all_df) == 0: continue

            gold_da = grid_data_to_xarray(
                all_df,
                Config.MASK_LON_MIN, Config.MASK_LON_MAX,
                Config.MASK_LAT_MIN, Config.MASK_LAT_MAX,
                Config.GRID_SPACING
            )

            gold_da_masked = mask_gold_region(
                gold_da,
                Config.INNER_MASK_LON_MIN, Config.INNER_MASK_LON_MAX,
                Config.INNER_MASK_LAT_MIN, Config.INNER_MASK_LAT_MAX
            )

            # === MIT 处理 (提前到这里，因为如果时间不匹配，就没必要存GOLD了) ===
            # 从文件名解析时间，而不是依赖还未保存的文件
            # 临时生成一个路径用于解析，或者直接用 parse_gold_time_from_path 解析 nh_file
            # 既然我们有 date_str (2023_001_20_10), 我们可以构造一个虚拟路径
            temp_gold_path = f"GOLD_NI1_gridded_data_{date_str}.nc"
            target_time = parse_gold_time_from_path(temp_gold_path)

            xr_mit_tec = load_mit_tec(
                target_time,
                (Config.MASK_LON_MIN, Config.MASK_LON_MAX),
                (Config.MASK_LAT_MIN, Config.MASK_LAT_MAX),
                Config.MIT_FILE_PATH,
                time_tolerance=timedelta(minutes=Config.TIME_TOLERANCE_MINUTES)  # 【关键传入】
            )

            # 只有当 MIT 加载成功 (没有报 ValueError) 后，才保存文件
            gold_save_path = os.path.join(Config.OUTPUT_DIR_GOLD, temp_gold_path)
            gold_da_masked.to_netcdf(gold_save_path)

            mit_save_path = os.path.join(Config.OUTPUT_DIR_MIT, f'MIT_TEC_data_{date_str}.nc')
            xr_mit_tec.to_netcdf(mit_save_path)

            processed_count += 1

        except ValueError as ve:
            # 专门捕获时间匹配失败
            # print(f"\n跳过 {date_str}: {ve}")
            skipped_count += 1
            continue
        except Exception as e:
            print(f"\nError processing {date_str}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ 处理结束！成功: {processed_count}, 跳过(时间不匹配): {skipped_count}")


if __name__ == "__main__":
    main()