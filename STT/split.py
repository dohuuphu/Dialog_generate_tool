
from asr_model.transcript import SubGenerator
from asr_model.model import ASRModel
from speechbrain.pretrained import EncoderASR
from asr_model.text_processing.inverse_normalize import InverseNormalizer
import glob
import os
normalizer = InverseNormalizer("vi")
asr_model = EncoderASR.from_hparams(source="/mnt/c/Users/phudh/Documents/source/label_TTS/config_model")


# files = glob.glob('/mnt/c/Users/phudh/Documents/data/mc_duco_2/*.mp3')
files = glob.glob('/mnt/c/Users/phudh/Documents/source/label_TTS/test.wav')
try:

    for file in files:
        print('111111111111111111111111',file)
        # path_file = "/mnt/c/Users/quangtd/Workplace/share/SpeechtoText/meo_xu/meoxu_story.mp3"
        result = SubGenerator(file, asr_model, normalizer, gector=None).transcript_split()
except:
    pass
