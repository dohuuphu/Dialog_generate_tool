
import shutil
import sys
import os
import torchaudio
from os.path import dirname, join, basename, exists

sys.path.append('/mnt/c/Users/phudh/Desktop/src/dialog_system/STT')
sys.path.append('/mnt/c/Users/phudh/Desktop/src/dialog_system/Identify_speaker')
from STT.speechbrain.pretrained import EncoderASR
from STT.asr_model.text_processing.inverse_normalize import InverseNormalizer
from STT.asr_model.audio import AudioFile
from STT.asr_model.variabels import *


from Identify_speaker.inference import Model as ID_model
from Identify_speaker.variables import *

from denoiser.denoiser.enhance_custom import Denoiser

import torch
import glob
import shutil
import numpy as np
import moviepy.editor as mp
from pydub import AudioSegment as am
from scipy.io import wavfile


AUDIO_ROOT = '/video/audio'
AUDIO_DENNOISE_ROOT = '/video/audio_denoise'
VIDEO_ROOT = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video'

class Dialog():
    def __init__(self) -> None:
        self.normalizer = InverseNormalizer("vi")
        self.asr_model = EncoderASR.from_hparams(source="/mnt/c/Users/phudh/Desktop/src/dialog_system/STT/config_model")
        self.id_model = ID_model()
        self.denoiser = Denoiser()
        self.count_speaker = 0

    def buffer_to_numpy(self, audio:AudioFile, buffer):
        if audio.wav_file.getsampwidth() == 1:
            # 8-Bit format is unsigned.
            datatype = np.uint8
            fconverter = lambda a : ((a / 255.0) - 0.5) * 2
        elif audio.wav_file.getsampwidth() == 2:
            # 16-Bit format is signed.'
            datatype = np.int16
            fconverter = lambda a : a / 32767.0
        
        signal = np.frombuffer(buffer, dtype=datatype)
        signal = fconverter(np.asarray(signal, dtype = np.float64))
        return signal

    def denoise_numpyData(self, audio_np):
        audio =  self.denoiser.enhance_numpyData(audio_np)
        return audio

    def inference(self, audio_path):
        trans_dict = None

        audio_ = AudioFile(audio_path)
        with open(audio_path.replace('.wav','.txt'), 'w') as w:
            for start, end, audio_buffer, s in audio_.split_file():
 
                # speech to text
                audio_np = self.buffer_to_numpy(audio_, audio_buffer)

                # audio_np = self.denoise_numpyData(audio_np).astype(np.float64)
                audio_t = torch.from_numpy(audio_np).unsqueeze(1)
                text = self.asr_model.transcribe_batch(self.asr_model.audio_normalizer(audio_t, SAMPLE_RATE).unsqueeze(0), torch.tensor([1.0]))[0] if self.asr_model else "emtpy"

                # ===========================================
                # tmp_path = 'test.wav'
                # wavfile.write(tmp_path, SAMPLE_RATE, audio_np.astype(np.float32) )
                # #format data
                # sound = am.from_file(tmp_path, format='wav')
                # sound = sound.set_frame_rate(SAMPLE_RATE)
                # sound = sound.set_channels(1)
                # sound.export(tmp_path, format='wav')
                # samplerate, data = wavfile.read(tmp_path)
                # audio_np = data.astype(np.float64)
                #=======================================

                # identify speaker
                if len(text.split(' ')) < 2:
                    continue
                speaker = 'Null'
                score = 0
                if self.id_model.database.num_speaker > 0:
                    stt, score, speaker, emb  = self.id_model.verify_speakers(audio_np)
                    if stt is IDENTIFIED:
                        info = f'speaker {speaker}'
                        if score >= 0.9:
                            self.save_speaker(emb, audio_np, speaker)
                    else: 
                        prefix = ''
                        if self.count_speaker == 2: # split dialog when meet speaker 3
                            self.id_model.database.clean_database()
                            self.count_speaker = 0
                            prefix = '====================================\n'

                        info = prefix + self.save_speaker(emb, audio_np)

                else:
                    emb = self.id_model.calculate_emb(audio_np)
                    info = self.save_speaker(emb, audio_np)

                i = f'{info} : {text} | pred {speaker} : {score:.3f} \n'
                w.writelines(i)
                print(i)

                
                

    
    def save_speaker(self, emb, audio, name = None):
        if name is None:
            self.count_speaker+=1
        if self.id_model.save_newEmb(emb = emb, name = name):
            info = f'New_speaker *{self.id_model.database.num_speaker}*'
            
            # save audio with emb
            name = (self.id_model.database.num_speaker) if name is None else name
            personalFolder = join(DB_ROOT, str(name))
            path_audio = join(personalFolder, f'{len(os.listdir(personalFolder))}.wav')
            wavfile.write(path_audio, SAMPLE_RATE, audio.astype(np.float32) )
        else:
            info = f'Save new speaker failed!!!'
        return info

    def extrac_audio_from_video(self, video_path):
        try:
            print(f'Extract audio from video ...')
            video_name = basename(video_path).split('.')[0]
            folder_path = join(VIDEO_ROOT, video_name)
            if exists(folder_path):
                try:
                    shutil.rmtree(folder_path)
                except OSError as e:
                    print("Error: %s : %s" % (folder_path, e.strerror))

            os.makedirs(folder_path)
            

            audio_path = join(folder_path, f'{video_name}.wav')
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)

            #format data
            sound = am.from_file(audio_path, format='wav')
            sound = sound.set_frame_rate(SAMPLE_RATE)
            sound = sound.set_channels(1)
            sound.export(audio_path, format='wav')

        except OSError as e:
            raise e

        return audio_path

    def denoise_fullAudio(self, audio_path):
        folder_path = '/'.join(audio_path.split('/')[0:-1])
        self.denoiser.enhance_fullAudio(folder_path, folder_path)
        return audio_path.replace('.wav', '_denoise.wav')


def main(files_path, dialog):
    [ shutil.rmtree(i) for i in glob.glob('/mnt/c/Users/phudh/Desktop/src/dialog_system/Identify_speaker/speaker_id/*')]
    

    # files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/hongnhan_tap21.mp4'
    
    # audio_path = dialog.extrac_audio_from_video(files_path)
    # audio_denoise_path = dialog.denoise_fullAudio(audio_path)

    
    dialog.id_model.database.clean_database()
    dialog.inference(files_path)

if __name__ == "__main__":
    [ shutil.rmtree(i) for i in glob.glob('/mnt/c/Users/phudh/Desktop/src/dialog_system/Identify_speaker/speaker_id/*')]
    dialog = Dialog()
    

    # files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/dialog/dialog.wav'
    # main(files_path, dialog)
    # files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/baongam/baongam_denoise.wav'
    # main(files_path, dialog)
    # files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/hongnhan_tap21/hongnhan_tap21_denoise.wav'
    # main(files_path, dialog)
    # files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/hồngnhan/hồngnhan_denoise.wav'
    # main(files_path, dialog)
    files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/phim1/phim1_denoise.wav'
    main(files_path, dialog)

    

    