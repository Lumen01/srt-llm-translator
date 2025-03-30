import logging
import sys
from .config import load_config

def setup_logger():
    # 从配置文件加载日志级别设置
    config = load_config()
    log_level = config.get('logging', {}).get('level', 'INFO').upper()

    # 创建日志记录器
    logger = logging.getLogger('srt_translator')
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # 避免日志重复
    if not logger.handlers:
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))

        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # 添加处理器到日志记录器
        logger.addHandler(console_handler)

    return logger

# 创建全局日志实例
logger = setup_logger()
