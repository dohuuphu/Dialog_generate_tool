#=======================================================
#Initialize model
#=======================================================

#import library
import copy
import imp
import torch
import torch
import os
import numpy as np
from collections import defaultdict
import sys
# sys.path.append("process/module/face_detection")
from FaceRecognition.config.config import config
from FaceRecognition.process.module.face_detection import retinaface, face_inference
from FaceRecognition.process.module.face_recognition.inference_face_embedding import get_face_embeded
import mxnet as mx
from collections import namedtuple

def init_models():
    print("------------------ Loading FaceRecognize model ----------------------")
    cuda = config["system"]["cuda"]
    gpu_id = 0 # Set GPU ID in config (os environment CUDA_VISIBLE_DEVICES)

    model_config = config["models"]
    
    print("Done load config relation, attribute")
    # Load model
    if cuda:

        if model_config["face_detection"]["active"]:
            model_detection = retinaface.RetinaFace(model_config["face_detection"]["model"], 0, 0, 'net3')
        else:
            model_detection = None
        
        if model_config["face_recognition"]["active"]:
            prefix = model_config["face_recognition"]["model"]
            sym, arg, aux = mx.model.load_checkpoint(prefix, 0)
            # define mxnet
            ctx = mx.gpu(gpu_id)
            mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
            mod.set_params(arg, aux)
            batch = namedtuple('Batch', ['data'])
            model_recognition = [mod, batch]
        else:
            model_recognition = None
            
        print("Done load face_recognition model")
    else:
        if model_config["face_detection"]["active"]:
            model_detection = retinaface.RetinaFace(model_config["face_detection"]["model"], 0, -1, 'net3')
        else:
            model_detection = None
        
        if model_config["face_recognition"]["active"]:
            prefix = model_config["face_recognition"]["model"]
            sym, arg, aux = mx.model.load_checkpoint(prefix, 0)
            # define mxnet
            ctx = mx.cpu()
            mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
            mod.set_params(arg, aux)
            batch = namedtuple('Batch', ['data'])
            model_recognition = [mod, batch]
        else:
            model_recognition = None
            
        print("Done load face_recognition model")
    # Warm up
    if model_config["face_recognition"]["active"]:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        for i in range(2):
            _ = get_face_embeded(img,model_recognition)
        print("Done warm up")
        
    return model_detection, model_recognition
