import time
import torch
import numpy as np
import librosa
import nemo.collections.asr as nemo_asr
from STT.asr_model.variabels import SAMPLE_RATE

from Identify_speaker.database_faiss import Database
from Identify_speaker.variables import *
# model storage: ~/.cache/torch/NeMo/NeMo_1.8.2/ecapa_tdnn/20b7839bda482a0b7d4b3390c337d2bc

class Model():
    def __init__(self):  
        self.database = Database()
        self.num_speaker = 0
        self.device = torch.device('cpu')
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='ecapa_tdnn', map_location= self.device)
        # self.speaker_model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder', device = 'cpu').eval()

    def verify_speakers(self, input_):
        start = time.time()
        cur_emb = self.calculate_emb(input_)

        print(f'cal_emb: {time.time() - start}')
        distance, id = self.database.storage.search(cur_emb, 1)        

        score = 1 - distance
        
        print(f'search speaker: {time.time() - start}')

        # Decision        
        stt = IDENTIFIED if score >= THRESHOLD else UN_IDENTIFIED

        return stt, float(score), int(id[0][0]), cur_emb

    def calculate_emb(self, input_):
        # input is the path
        if type(input_) is str: 
            cur_emb = self.speaker_model.get_embedding(input_)

        # input is the data type
        else: 
            cur_emb = self.preprocess_audio(input_)

        return cur_emb.unsqueeze(0).cpu().numpy()

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

        # embs = embs / torch.linalg.norm(embs)

        del audio_signal, audio_signal_len
        return embs.squeeze(0)
                    
    def save_newEmb(self, root_folder, name = None, audio = None, emb = None):

        if not audio and emb is None:
            return False, None

        emb_ = self.calculate_emb(audio) if emb is None else emb

        return self.database.save_spkEmb(root_folder, emb_, name)

            
if __name__ == "__main__":
    # audio = '/mnt/c/Users/phudh/Desktop/src/dialog_system/dialog.m4a'
    model = Model()
    data, sr = librosa.load(audio, sr=None)
    print(model.verify_speakers(data))