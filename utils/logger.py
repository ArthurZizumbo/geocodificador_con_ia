# src/geocodificador_ine_mvp/utils/logger.py
import logging
from pythonjsonlogger import jsonlogger

def setup_logger():
    logger = logging.getLogger("geocodificador")
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter() # type: ignore
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger