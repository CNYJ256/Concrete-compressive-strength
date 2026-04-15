"""
日志工具模块。

目标：
- 统一日志格式，便于排查数据对齐、维度错误、训练异常等问题。
"""

import logging
from typing import Optional


def get_logger(name: str, level: int = logging.INFO, fmt: Optional[str] = None) -> logging.Logger:
    """
    创建并返回一个日志对象。

    参数：
    - name: 日志器名称，通常传 __name__。
    - level: 日志级别，默认 INFO。
    - fmt: 自定义日志格式；若不传则使用默认格式。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加 Handler（在多次 import 或重复运行脚本时很常见）
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt or "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
