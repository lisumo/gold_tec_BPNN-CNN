# config.py
import os


class Config:
    # =========================================
    # 1. 基础路径与环境判断
    # =========================================
    RAW_GOLD_PATHS = [
        "/mnt/public/GOLD/2023/spdf.gsfc.nasa.gov/pub/data/gold/level1c/2023",
        "/mnt/public/GOLD/2024/spdf.gsfc.nasa.gov/pub/data/gold/level1c/2024"
    ]
    LOCAL_GOLD_PATH = "E:/research/TEC/2023/"

    SERVER_MIT_PATH = "/mnt/geodata/GIM/mit_obj/mitg2023_2024.obj"
    LOCAL_MIT_PATH = "E:/research/TEC/mitg2023_2024.obj"

    if os.path.exists(RAW_GOLD_PATHS[0]):
        print(f"🚀 [Config] 服务器环境 (跨年数据源)")
        # 这里的 GOLD_PATHS 主要给 loader 使用，指向预处理后的目录(output)或者原始目录
        # 但在 02_build_dataset 中我们已经改为优先读 OUTPUT_DIR_GOLD
        MIT_FILE_PATH = SERVER_MIT_PATH if os.path.exists(SERVER_MIT_PATH) else LOCAL_MIT_PATH

        # 【修复报错】必须定义 HAS_DOY_SUBFOLDERS
        HAS_DOY_SUBFOLDERS = True
    else:
        print(f"💻 [Config] 本地环境")
        RAW_GOLD_PATHS = [LOCAL_GOLD_PATH]  # 本地也转为列表统一处理
        MIT_FILE_PATH = LOCAL_MIT_PATH

        # 【修复报错】本地环境通常是扁平的
        HAS_DOY_SUBFOLDERS = False

    # =========================================
    # 2. 关键匹配参数 (严格匹配模式)
    # =========================================

    # 周期性窗口划分开关 (True=启用; False=原有随机划分)
    USE_PERIODIC_SPLIT = True

    # 周期性划分参数 (单位: 天)
    # 逻辑: (Current_DOY - Start_DOY) % window_size
    PERIODIC_SPLIT_PARAMS = {
        "window_size": 50,  # 窗口总长 50 天
        "train_days": 35,  # 第 1-35 天 -> 训练集
        "val_days": 10,  # 第 36-45 天 -> 验证集
        "test_days": 5  # 第 45-50 天 -> 测试集
    }

    # 是否开启陆地遮挡模式
    ENABLE_LAND_MASKING = False

    # 定义遮挡区域
    LAND_MASK_REGION = {
        'lon_min': -60,
        'lon_max': -50,
        'lat_min': -30,
        'lat_max': -10
    }

    # GOLD 数据筛选
    START_YEAR = 2023
    START_DOY = 1
    TOTAL_DAYS = 350
    FIXED_SUFFIX = ""

    # 严格的时间表
    """TIME_PAIRS = [
        ("00_10", "00_10"), ("00_22", "00_25"),
        ("20_10", "20_10"), ("20_22", "20_25"), ("20_40", "20_40"), ("20_52", "20_55"),
        ("21_10", "21_10"), ("21_22", "21_25"), ("21_40", "21_40"), ("21_52", "21_55"),
        ("22_10", "22_10"), ("22_22", "22_25"), ("22_40", "22_40"), ("22_52", "22_55"),
        ("23_10", "23_10"), ("23_22", "23_25"), ("23_40", "23_40"), ("23_52", "23_55")
    ]"""
    TIME_PAIRS = [
        ("00_10", "00_10"),
        ("20_10", "20_10"),
        ("21_10", "21_10"),
        ("22_10", "22_10"),
        ("23_10", "23_10"),
    ]

    # 中间结果输出路径
    BASE_OUTPUT_DIR = 'GOLD_MIT_Results'
    OUTPUT_DIR_IMAGE = os.path.join(BASE_OUTPUT_DIR, 'Images')
    OUTPUT_DIR_GOLD = os.path.join(BASE_OUTPUT_DIR, 'GOLD_NC')
    OUTPUT_DIR_MIT = os.path.join(BASE_OUTPUT_DIR, 'MIT_NC')
    OUTPUT_DIR_COMPARE = os.path.join(BASE_OUTPUT_DIR, 'Comparison_Plots')
    NC_LIST_FILENAME = "NCFilesList.txt"

    # 最终数据集与模型路径
    DATASET_PATH = 'dataset/tec_2d_enhanced_dataset.pth'
    MODEL_SAVE_PATH = 'tec_2d_fusion_cnn_bpnn_model.pth'

    # =========================================
    # 2. 数据处理配置 (Data Processing)
    # =========================================

    # 网格化参数
    GRID_SPACING = 1.0  # 经纬度格网间隔

    # 区域掩码范围
    MASK_LON_MIN = -90
    MASK_LON_MAX = -20
    MASK_LAT_MIN = -40
    MASK_LAT_MAX = 40

    # 内部掩码去除
    INNER_MASK_LON_MIN = -90
    INNER_MASK_LON_MAX = -75
    INNER_MASK_LAT_MIN = 30
    INNER_MASK_LAT_MAX = 40

    # 数据集构建参数
    TIME_TOLERANCE_MINUTES = 15
    MAX_MISSING_RATIO = 0.45
    USE_INTERPOLATION = True

    # =========================================
    # 3. 模型超参数 (Model Hyperparameters)
    # =========================================
    MODEL_PARAMS = {
        'model_mode': 'fusion',  # 可选: 'fusion', 'cnn', 'bpnn'

        'input_shape': None,
        'input_channels': None,
        'time_features_dim': None,

        # 数据集划分 (仅当 USE_PERIODIC_SPLIT=False 时生效)
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42,

        # 网络结构
        'conv_channels': [64, 128, 256],
        'kernel_sizes': [3, 3, 5],
        'time_dense_layers': [64, 32, 16],
        'spatial_dense_layers': [128, 64, 32],
        'fusion_dense_layers': [256, 128],
        'bpnn_hidden_layers': [64, 32],
        'dropout_rate': 0.1,

        # 特殊机制
        'use_spatial_attention': True,
        'attention_kernel_size': 7,
        'attention_scale': 1,
        'use_bpnn_background': True,
        'fusion_weight': 0.6,

        # 平滑损失配置 (Smoothness Loss)
        'enable_smooth_loss': True,  # 开关：是否启用平滑损失
        'smooth_loss_weight': 0.02,  # 权重：建议 0.01 ~ 0.1

        # 训练参数
        'learning_rate': 0.0015,
        'weight_decay': 1e-5,
        'loss_beta': 2.5,  # SmoothL1Loss beta
        'epochs': 1000,
        'batch_size': 64,
        'early_stopping_patience': 50,
        'print_interval': 10
    }