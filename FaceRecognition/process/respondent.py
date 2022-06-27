import cv2
import pytz
import json
import numpy as np
from PIL import Image
from datetime import datetime
from config.init_models import init_models
from process.face_recognition import face_recognition

class Respondent:
    def __init__(self):
        self.list_models = init_models()

    def response(self, url):
        if url is None:
            print("This is a bug")
            return "URL không hợp lệ vui lòng thử lại"
        # print(self.list_models)
        face_recognition(url, self.list_models)
        
        # self.save_result_to_storage(image, graph_result)
        
        # return graph_result
        
    
    # def save_result_to_storage(self, image, graph_result):
    #     cv2.imwrite("{}/{}".format(origin_path, "{}.jpg".format(self.image_id)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
    #     f_graph_info = open("{}/{}".format(graph_info_path, "{}.json".format(self.image_id)), "w", encoding="utf-8")
    #     json.dump(graph_result, f_graph_info, indent=2)
    #     f_graph_info.close()
        
        
respondent = Respondent()
