import time
import torch
import numpy as np
import librosa
import nemo.collections.asr as nemo_asr
from STT.asr_model.variabels import SAMPLE_RATE

from database_faiss import Database
from variables import *
# model storage: ~/.cache/torch/NeMo/NeMo_1.8.2/ecapa_tdnn/20b7839bda482a0b7d4b3390c337d2bc

class Model():
    def __init__(self):        
        self.database = {'1':[], '2':[]}
        self.num_speaker = 0
        self.device = torch.device('cpu')
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='ecapa_tdnn', map_location= self.device)
        # self.speaker_model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder', device = 'cpu').eval()

    def verify_speakers(self, input_):
        start = time.time()
        X = self.calculate_emb(input_)

        print(f'cal_emb: {time.time() - start}')
        pred = [0,'null']
        for id in self.database:
            for Y in self.database[id]:

                # Score
                similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
                similarity_score = (similarity_score + 1) / 2
                if similarity_score > pred[0]:
                    pred = [similarity_score, id]

        
        print(f'search speaker: {time.time() - start}')

        # Decision        
        stt = IDENTIFIED if pred[0] >= THRESHOLD else UN_IDENTIFIED
        return stt, float(pred[0]), pred[1], X

    def calculate_emb(self, input_):
        if type(input_) is str: # path type 
            cur_emb = self.speaker_model.get_embedding(input_)
        else: # data type
            cur_emb = self.preprocess_audio(input_)

        return cur_emb

    def preprocess_audio(self, data):
        # ndarray type
        if type(data) is np.ndarray:
            audio = data.astype(np.float32)
            audio_length = data.shape[0]
            
        # buffer type
        else: 
            audio = np.array(data, dtype='f')
            audio_length = data.shape[0]
            
        audio_signal, audio_signal_len = (
            torch.tensor([audio], device=self.device),
            torch.tensor([audio_length], device=self.device),
        )
        self.speaker_model.freeze()

        _, embs = self.speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

        embs = embs / torch.linalg.norm(embs)

        del audio_signal, audio_signal_len
        return embs.squeeze(0)
                    
    def save_newEmb(self, name = None, audio = None, emb = None):

        if not audio and emb is None:
            return False

        emb_ = self.calculate_emb(audio) if emb is None else emb

        self.database[name].append(emb_)

        return True

            
if __name__ == "__main__":
    # audio = '/mnt/c/Users/phudh/Desktop/src/dialog_system/dialog.m4a'
    model = Model()
    data, sr = librosa.load(audio, sr=None)
    print(model.verify_speakers(data))