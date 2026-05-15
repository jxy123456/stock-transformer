import sys
from pathlib import Path

from loguru import logger


def setup_logger(name: str, log_dir: str = None, level: str = "INFO"):
    log_dir_path = Path(log_dir) if log_dir else Path(__file__).parent.parent / "outputs" / "logs"
    log_dir_path.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
    )

    logger.add(
        log_dir_path / f"{name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )

    return logger
