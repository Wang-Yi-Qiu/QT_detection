import logging

from .config import APP_LOG_FILE, LOG_DIR


def get_app_logger() -> logging.Logger:
    logger = logging.getLogger("yolo_qt_app")
    if logger.handlers:
        return logger

    LOG_DIR.mkdir(exist_ok=True)
    handler = logging.FileHandler(APP_LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
