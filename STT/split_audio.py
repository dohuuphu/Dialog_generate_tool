
from speechbrain.pretrained import EncoderASR
from asr_model.text_processing.inverse_normalize import InverseNormalizer
from asr_model.audio import AudioFile
import torchaudio
import torch
import numpy as np
from asr_model.variabels import *

files_path = '/mnt/c/Users/phudh/Documents/source/label_TTS/test.wav'
class Dialog():
    def __init__(self) -> None:
        self.normalizer = InverseNormalizer("vi")
        self.asr_model = EncoderASR.from_hparams(source="/mnt/c/Users/phudh/Documents/source/label_TTS/config_model")

    def buffer_to_numpy(self, audio:AudioFile, buffer):
        datatype = None

        if audio.wav_file.getsampwidth() == 1:
            # 8-Bit format is unsigned.
            datatype = np.uint8
            fconverter = lambda a : ((a / 255.0) - 0.5) * 2
        elif audio.wav_file.getsampwidth() == 2:
            # 16-Bit format is signed.
            datatype = np.int16
            fconverter = lambda a : a / 32767.0
        
        signal = np.frombuffer(buffer, dtype=datatype)
        signal = fconverter(np.asarray(signal, dtype = np.float64))
        return signal


    def inference(self, audio_path):

        trans_dict = None
        audio_ = AudioFile(audio_path)
        for start, end, audio_buffer, s in audio_.split_file():
            if trans_dict is not None:
                if start - trans_dict.get('end', 0) > self.split_threshold or len(trans_dict['tokens']) > self.max_len:
                    final_transcript = trans_dict['tokens']
        
            signal = torch.from_numpy(self.buffer_to_numpy(audio_, audio_buffer)).unsqueeze(1)
            
            tokens = self.asr_model.transcribe_batch(self.asr_model.audio_normalizer(signal, SAMPLE_RATE).unsqueeze(0), torch.tensor([1.0]))[0]
            print('tokens ', tokens)

if __name__ == "__main__":
    sys = Dialog()
    sys.inference(files_path)