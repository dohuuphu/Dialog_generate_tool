import time
import torch
import numpy as np
import librosa
import nemo.collections.asr as nemo_asr
from STT.asr_model.variabels import SAMPLE_RATE

from database_faiss import Database
from variables import *


import glob
from os.path import join, dirname

import sys
sys.path.append('./deep_speaker')

from deep_speaker.audio import read_mfcc, mfcc_fbank
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

class Model():
    def __init__(self):        
        self.database = Database()
        self.device = torch.device('cpu')
        self.speaker_model = DeepSpeakerModel()
        self.speaker_model.m.load_weights('/mnt/c/Users/phudh/Desktop/src/dialog_system/Identify_speaker/deep_speaker/ResCNN_triplet_training_checkpoint_265.h5', by_name=True)
        self.num_speaker = 0

    def verify_speakers(self, input_):
        start = time.time()
        cur_emb = self.calculate_emb(input_)
        print(f'cal_emb: {time.time() - start}')

        pred = [0, 'Null']
        for speaker_emb in glob.glob('/mnt/c/Users/phudh/Desktop/src/dialog_system/Identify_speaker/speaker_id/*/*.txt'):
            speaker_name = dirname(speaker_emb).split('/')[-1]
            ref_emb = self.calculate_emb(speaker_emb)

            # Compute the cosine similarity
            score = batch_cosine_similarity(cur_emb, ref_emb)
            
            if score > pred[0]:
                pred = [score, speaker_name]

        print(f'search speaker: {time.time() - start}')

        # Decision        
        stt = IDENTIFIED if pred[0] >= THRESHOLD else UN_IDENTIFIED
        return stt, float(pred[0]), pred[1], cur_emb

    def calculate_emb(self, input_):
        if type(input_) is str: # path type 
            mfcc = sample_from_mfcc(read_mfcc('/mnt/c/Users/phudh/Desktop/src/dialog_system/audio/Phu_bt.wav', DEFAULT_RATE), 160)
            cur_emb = self.speaker_model.m.predict(np.expand_dims(mfcc, axis=0))
        else: # data type
            cur_emb = self.preprocess_audio(input_)

        return cur_emb

    def preprocess_audio(self, data):
        # ndarray type
        if type(data) is np.ndarray:
            audio = data.astype(np.float32)
            energy = np.abs(audio)
            silence_threshold = np.percentile(energy, 95)
            offsets = np.where(energy > silence_threshold)[0]
            # left_blank_duration_ms = (1000.0 * offsets[0]) // self.sample_rate  # frame_id to duration (ms)
            # right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // self.sample_rate
            # TODO: could use trim_silence() here or a better VAD.
            audio_voice_only = audio[offsets[0]:offsets[-1]]
            mfcc = sample_from_mfcc(mfcc_fbank(audio_voice_only, DEFAULT_RATE), NUM_FRAMES)

            return self.speaker_model.m.predict(np.expand_dims(mfcc, axis=0))

                    
    def save_newEmb(self, name = None, audio = None, emb = None):

        if not audio and emb is None:
            return False

        emb_ = self.calculate_emb(audio) if emb is None else emb

        return self.database.save_spkEmb(emb_, name)

            
if __name__ == "__main__":
    # audio = '/mnt/c/Users/phudh/Desktop/src/dialog_system/dialog.m4a'
    model = Model()
    data, sr = librosa.load(audio, sr=None)
    print(model.verify_speakers(data))