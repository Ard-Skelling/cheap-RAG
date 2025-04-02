import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


# Local modules
from configs.config import LOGGER_CONFIG


BASE_PATH = Path(__file__).parent.parent.parent


# Log path
log_dir = BASE_PATH.joinpath('logs')
log_dir.mkdir(exist_ok=True, parents=True)
log_file = str(log_dir.joinpath(LOGGER_CONFIG.file_name))

# Initiate logger
logger = logging.getLogger(LOGGER_CONFIG.name)
logger.setLevel(getattr(logging, LOGGER_CONFIG.level))

# Set TimedRotatingFileHandler
handler = TimedRotatingFileHandler(
    log_file, 
    when=LOGGER_CONFIG.when, 
    interval=LOGGER_CONFIG.interval, 
    backupCount=LOGGER_CONFIG.backup_count,
    delay=LOGGER_CONFIG.delay,
    utc=LOGGER_CONFIG.utc
)
handler.suffix = "%Y-%m-%d.log"
handler.setFormatter(logging.Formatter('%(asctime)s -  %(levelname)s - %(module)s - %(funcName)s - %(message)s'))
logger.addHandler(handler)

consol_handler = logging.StreamHandler(sys.stdout)
consol_handler.setFormatter(logging.Formatter('%(asctime)s -  %(levelname)s - %(module)s - %(funcName)s - %(message)s'))
logger.addHandler(consol_handler)
