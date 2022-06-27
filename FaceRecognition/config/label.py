import json

from config.config import config
import utils
from utils.load_faiss import load_db


face_database = load_db(config["face_database_path"], use_gpu=False)


