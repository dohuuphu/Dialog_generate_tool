
import sys
import time
import cv2 as cv
import numpy as np
sys.path.append("/mnt/c/Users/phudh/Desktop/src/dialog_system/FaceRecognition/process/module/face_detection")

from FaceRecognition.config.init_models import init_models
from FaceRecognition.process.module.face_detection.face_inference import get_face_area
from FaceRecognition.process.module.face_recognition.inference_face_embedding import get_face_embeded

from FaceRecognition.utils.database_faiss import Database
from FaceRecognition.variables import *

from sklearn import preprocessing

THRES_HOLDE =  0.8


class FaceRecognize():
    def __init__(self):
        self.f_detection, self.f_recognition = init_models()
        self.database = Database()

    
    # return list of faces
    ''' Input: full image (np.ndarray)'''
    def get_face(self, image, crop_img = True, img_raw = None):
        imga_crop, mouth_crop, _ = get_face_area(image, self.f_detection, THRES_HOLDE, crop_img = crop_img, img_raw = img_raw)

        return imga_crop, mouth_crop

    ''' Input: face image (np.ndarray)'''
    def get_faceEmb(self, face):
        return preprocessing.normalize(np.expand_dims(get_face_embeded(face, self.f_recognition), axis=0), norm='l2')

    def verify_face(self, face):
        start = time.time()

        emb = self.get_faceEmb(face)
        # emb = np.expand_dims(emb, axis=0)

        score, id = self.database.storage.search(emb, 1)        

        # print(f'search face: {time.time() - start}')

        # Decision        
        stt = IDENTIFIED if score >= THRESHOLD and score <=1 else UN_IDENTIFIED
        return stt, float(score), id[0][0], emb

    def save_newEmb(self, root_folder, name = None, emb = None, prefix = ''):
        ''' save new speaker, create new speaker if name is NONE'''
        if emb is None:
            return False, None
        else:
            return self.database.save_spkEmb(root_folder, emb, name, prefix)
        


if __name__ == '__main__':
    face = FaceRecognize()
    PATH = '/mnt/c/Users/phudh/Desktop/src/dialog_system/draf/a.png'
    img = cv.imread(PATH)
    print(type(img))
    print(len(face.get_face(img)))