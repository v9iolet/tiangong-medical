import sys
from pathlib import Path
from loguru import logger
from src.core.config import get_settings


# 项目一启动，必须自己调用这个方法，配置好日志组件的工作规则
def setup_logger() -> None:
    settings = get_settings()
    logger.remove()

    # 控制台
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 文件
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_dir / "{time:YYYY-MM-DD}.log"),
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
    )

    # 审计日志：只收 logger.bind(audit=True) 的记录，单独落盘并保留更久，便于事后追溯
    # 普通 sink 不做过滤，审计记录仍会同时出现在控制台和业务日志中
    logger.add(
        str(log_dir / "audit_{time:YYYY-MM-DD}.log"),
        level="INFO",
        filter=lambda record: record["extra"].get("audit", False),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="180 days",
        compression="gz",
        encoding="utf-8",
    )