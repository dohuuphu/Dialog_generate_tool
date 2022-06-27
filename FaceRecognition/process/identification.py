import sys
sys.path.append("process/module/face_detection")
from process.module.face_detection import face_inference
from process.module.face_recognition.inference_face_embedding import get_face_embeded
from config.label import face_database
from sklearn import preprocessing
from config.constant import EMBEDDING_DIMENSION

def identification(image, model_detection, model_recognition, threshold_detect, threshold_recog):
    list_len_embedding, list_person_name, index_faiss = face_database
    result = []
    height, width, _ = image.shape
    scales = [720, 1280]
    crop_faces, box_info, landmarks = face_inference.get_face_area(image, model_detection, threshold_detect, scales)
    for i in range(len(crop_faces)):
        face = crop_faces[i]
        bounding_box = list(map(int,box_info[i][0:4]))
        face_embeded = get_face_embeded(face, model_recognition)
        print(face_embeded.shape)
        xq = face_embeded.astype('float32').reshape(1, EMBEDDING_DIMENSION)
        xq = preprocessing.normalize(xq, norm='l2')
        distances, indices = index_faiss.search(xq, 1)
        
        position = indices[0][0]
        sum = 0
        for idx in range(len(list_person_name)):
            sum += list_len_embedding[idx]
            if position < sum:
                if distances[0][0] >= threshold_recog:
                    result.append([list_person_name[idx], bounding_box, landmarks[i], distances[0][0],crop_faces[i]])
                else:
                    result.append(["stranger", bounding_box, landmarks[i], distances[0][0],crop_faces[i]])
                break
    return result