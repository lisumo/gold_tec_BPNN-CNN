# src/utils/logger.py
import logging
import os
import sys
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, exp_name):
        # 创建日志目录 logs/exp_name/
        self.exp_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # 1. 设置文本日志 (logging)
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # 文件处理器 (写入 train_log.txt)
        file_path = os.path.join(self.exp_dir, 'train_log.txt')
        file_handler = logging.FileHandler(file_path, mode='a')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 控制台处理器 (输出到屏幕)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 2. 设置 TensorBoard
        # TensorBoard 文件保存在 logs/exp_name/events/
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'events'))

        self.log(f"日志系统初始化完成。日志目录: {self.exp_dir}")

    def log(self, message):
        """记录文本日志"""
        self.logger.info(message)

    def log_metric(self, tag, value, step):
        """记录数值到 TensorBoard"""
        self.writer.add_scalar(tag, value, step)

    def get_save_dir(self):
        """获取保存模型的目录"""
        return self.exp_dir

    def close(self):
        self.writer.close()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)