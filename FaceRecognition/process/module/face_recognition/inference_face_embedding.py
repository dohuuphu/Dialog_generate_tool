import numpy as np
import cv2
import mxnet as mx

# convert array
def get_array(face_chip):
    face_chip = cv2.cvtColor(face_chip, cv2.COLOR_BGR2RGB)
    face_chip = face_chip.transpose(2, 0, 1)
    face_chip = face_chip[np.newaxis, :] # 4d
    array = mx.nd.array(face_chip)
    return array


def get_face_embeded(img, model_recognition):
    mod, batch = model_recognition
    
    array = get_array(img)
    mod.forward(batch([array]))
    feat = mod.get_outputs()[0].asnumpy()

    return feat[0]
