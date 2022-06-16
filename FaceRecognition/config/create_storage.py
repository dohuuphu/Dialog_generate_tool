import logging
import os

from config.config import config


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    if (l.hasHandlers()):
        l.handlers.clear()
    l.setLevel(level)
    l.addHandler(fileHandler) 


base_storage = config["storage"]["path"]

# KVQA
result_config = config["storage"]["result"]
result_path = result_config["path"]
result_image_path, face_info_path = [os.path.join(result_path, pth) for pth in result_config["sub_path"]]
log_file = os.path.join(result_path, result_config["log"])

if (not os.path.isdir(base_storage)):
    os.mkdir(base_storage)

if (not os.path.isdir(result_path)):
    os.mkdir(result_path)
if (not os.path.isdir(result_image_path)):
    os.mkdir(result_image_path)
if (not os.path.isdir(face_info_path)):
    os.mkdir(face_info_path)


setup_logger('result_face', log_file)
logger_general = logging.getLogger('log_face')
