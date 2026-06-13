"""
日志工具模块
提供统一的日志配置和使用接口
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(
    name: str = "qlib",
    log_file: str = None,
    level: str = "INFO",
    console: bool = True,
    rotation: bool = True,
) -> logging.Logger:
    """
    设置日志记录器

    Parameters
    ----------
    name : str
        日志记录器名称
    log_file : str
        日志文件路径
    level : str
        日志级别 DEBUG / INFO / WARNING / ERROR
    console : bool
        是否输出到控制台
    rotation : bool
        是否使用日志轮转

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if rotation:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8"
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "qlib") -> logging.Logger:
    """获取日志记录器"""
    return logging.getLogger(name)


class TradeLogger:
    """交易日志记录器"""

    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir or "./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 交易日志
        self.trade_logger = setup_logger(
            "trade",
            log_file=str(self.log_dir / "trade.log"),
            console=False
        )

        # 错误日志
        self.error_logger = setup_logger(
            "error",
            log_file=str(self.log_dir / "error.log"),
            level="ERROR",
            console=False
        )

    def log_trade(self, action: str, symbol: str, price: float,
                  shares: int, amount: float, timestamp: str = None):
        """记录交易"""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{action} {symbol} {shares}股 @{price:.2f} 金额:{amount:.2f}"
        self.trade_logger.info(f"[{ts}] {msg}")

    def log_error(self, error: str, exc_info: bool = False):
        """记录错误"""
        self.error_logger.error(error, exc_info=exc_info)

    def log_signal(self, date: str, symbols: list):
        """记录选股信号"""
        self.trade_logger.info(f"[{date}] 选股: {', '.join(symbols)}")

    def log_rebalance(self, date: str, old_pos: set, new_pos: set):
        """记录调仓"""
        to_sell = old_pos - new_pos
        to_buy = new_pos - old_pos
        self.trade_logger.info(
            f"[{date}] 调仓: 卖出{len(to_sell)}只, 买入{len(to_buy)}只"
        )
