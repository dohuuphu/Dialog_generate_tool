
import shutil
import sys
import os
import torchaudio
import cv2 as cv
from os.path import dirname, join, basename, exists

sys.path.append('/mnt/c/Users/phudh/Desktop/src/dialog_system/STT')
# sys.path.append('/mnt/c/Users/phudh/Desktop/src/dialog_system/Identify_speaker')
# sys.path.append('/mnt/c/Users/phudh/Desktop/src/dialog_system/FaceRecognition')
# sys.path.append('/mnt/c/Users/phudh/Desktop/src/dialog_system/FaceRecognition/process/module/face_detection')
from STT.speechbrain.pretrained import EncoderASR
from STT.asr_model.text_processing.inverse_normalize import InverseNormalizer
from STT.asr_model.audio import AudioFile
from STT.asr_model.variabels import *


from Identify_speaker.inference import Model as ID_model
from Identify_speaker.variables import *


from FaceRecognition.face_module import FaceRecognize

from denoiser.denoiser.enhance_custom import Denoiser

from moviepy.editor import *

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
        self.asr_model = None#EncoderASR.from_hparams(source="/mnt/c/Users/phudh/Desktop/src/dialog_system/STT/config_model")
        self.id_model = ID_model()
        self.denoiser = Denoiser()
        self.faceRecognize = FaceRecognize()
        self.video_name = None
        self.resultSpace = None
        self.cv_resultSpace = None
        self.id_faceFolder = None
        self.asr_resultSpace = None
        self.id_speakerFolder = None
        self.dialog_timeseries = []
        self.count_speaker = 0
        self.count_draf =0

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

    def inference(self, audio_path):
        self.count_speaker = 0
        # id_speakerFolder = join(self.asr_resultSpace, 'id_speaker')
        audio_ = AudioFile(audio_path)
        with open(audio_path.replace('.wav','.txt'), 'w') as w:
            for start, end, audio_buffer, s in audio_.split_file():
 
                # speech to text
                audio_np = self.buffer_to_numpy(audio_, audio_buffer)

                # audio_np = self.denoise_numpyData(audio_np).astype(np.float64)
                audio_t = torch.from_numpy(audio_np).unsqueeze(1)
                text = self.asr_model.transcribe_batch(self.asr_model.audio_normalizer(audio_t, SAMPLE_RATE).unsqueeze(0), torch.tensor([1.0]))[0] if self.asr_model else "emtpy a v c"

                self.count_draf+=1
                path_audio = f'/mnt/c/Users/phudh/Desktop/src/dialog_system/draf/{self.count_draf}.wav'
                wavfile.write(path_audio, SAMPLE_RATE, audio_np.astype(np.float32) )


                # identify speaker
                if len(text.split(' ')) < 3:
                    continue
                speaker = 'Null'
                score = 0

                stt, score, speaker_id, emb  = self.id_model.verify_speakers(audio_np)
                if stt is IDENTIFIED:
                    speaker = self.id_model.database.map_storage[speaker_id]
                    info = f'speaker {speaker}'
                    if score >= 0.9:
                        self.save_speaker(self.id_speakerFolder, emb, audio_np, speaker)
                else: 
                    prefix = ''
                    # if self.count_speaker == 2: # split dialog when meet speaker 3
                    #     # self.id_model.database.clean_database()
                    #     # self.save_emb()
                    #     self.id_model.database = {'1':[], '2':[]}
                    #     self.count_speaker = 0
                    #     prefix = '====================================\n'

                    info = prefix + self.save_speaker(self.id_speakerFolder, emb, audio_np)

                # else:
                #     emb = self.id_model.calculate_emb(audio_np)
                #     info = self.save_speaker(emb, audio_np)

                i = f'{info} : {text} | pred {speaker} : {score:.3f} \n'

                self.dialog_timeseries.append([start/1000, end/1000, text])
                w.writelines(i)
                print(i)
        
        self.identify_speaker_inVideo()
   
    def cut_person_inVideo(self):
        ''' Cut sub_clip depending on audio segment'''
        result_folder = join(self.cv_resultSpace, 'face_video')
        self.create_folder(result_folder)
        # video_path = join(VIDEO_ROOT, f'{self.video_name}.mp4')
        try:
            video = VideoFileClip(video_path)
            for idx, sentence in enumerate(self.dialog_timeseries):
                start_time, end_time, _ = sentence
                cutting = video.subclip(start_time, end_time)
                cutting.write_videofile(f"{join(result_folder, str(idx))}.mp4")
                
            return result_folder

        except OSError as e:
            raise e

    
    def identify_speaker_inVideo(self):
        subClip_path = self.cut_person_inVideo()

        with open(join(self.cv_resultSpace, 'face_recognize.txt'), 'w') as wr:
            for subClip_path_ in glob.glob(f'{subClip_path}/*'):

                name_video = subClip_path_.split('/')[-1].split('.')[0]
                # face_inSubclip = join(self.id_faceFolder, name_video)
                # self.create_folder(face_inSubclip)

                try:
                    cap = cv.VideoCapture(subClip_path_)
                    if (cap.isOpened()== False):
                        print(f"Error opening {subClip_path_}")
                    frame_index = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        frame_index += 1

                        if frame_index % 10 == 1: continue
                        if ret:
                            faces = self.faceRecognize.get_face(frame)
                            for face in faces:
                                stt, score, face_name_id, emb = self.faceRecognize.verify_face(face)

                                face_name = self.faceRecognize.database.map_storage[face_name_id] if len(self.faceRecognize.database.map_storage) > 0 else 'Null'
                                if stt is IDENTIFIED:
                                    
                                    info = f'face {face_name}'
                                    # if score >= 0.9:
                                    _ = self.save_face(self.id_faceFolder, emb, face, face_name, prefix = name_video)
                                else: 
                                    info = self.save_face(self.id_faceFolder, emb, face, prefix = name_video)

                                i = f'{info} | pred {face_name} : {score:.3f} \n'
                                wr.writelines(i)
                                print(i)
                        else:
                            break

                except OSError as e:
                    # print(e)s
                    raise e


    def save_speaker(self, root_folder, emb, audio, name = None, ):
        ''' save embedded feature of speaker'''   
        if name is None:
            self.count_speaker+=1
            name = str(self.count_speaker)

        success, emb_path = self.id_model.save_newEmb( root_folder, emb = emb, name = name)

        # save audio with emb
        wavfile.write(emb_path.replace('.txt', '.wav'), SAMPLE_RATE, audio.astype(np.float32) )

        return f'New_speaker *{self.count_speaker}*' if success else f'Save new speaker failed!!!'

    def save_face(self, root_folder, emb, face, name = None, prefix = ''):
        ''' save embedded feature of face'''   

        success, emb_path =  self.faceRecognize.save_newEmb(root_folder, name, emb, prefix)
        
        cv.imwrite(emb_path.replace('.txt', '.jpg'), face)

        return f'New_face *{self.faceRecognize.database.num_speaker}*' if success else f'Save new speaker failed!!!'

    def extrac_audio_from_video(self, video_path):
        try:
            print(f'Extract audio from video ...')

            # extract audio
            audio_path = join(self.asr_resultSpace, f'{self.video_name}.wav')
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)

            # format audio
            sound = am.from_file(audio_path, format='wav')
            sound = sound.set_frame_rate(SAMPLE_RATE)
            sound = sound.set_channels(1)
            sound.export(audio_path, format='wav')

        except OSError as e:
            raise e

        return audio_path

    def denoise_fullAudio(self, audio_path):
        ''' denoise audio with path'''
        folder_path = '/'.join(audio_path.split('/')[0:-1])
        self.denoiser.enhance_fullAudio(folder_path, folder_path)
        return audio_path.replace('.wav', '_denoise.wav')

    def denoise_numpyData(self, audio_np):
        ''' denoise audio with np.ndarray type'''
        audio =  self.denoiser.enhance_numpyData(audio_np)
        return audio

    def create_folder(self, folder_path):
        ''' create folder with path, remove old folder if it exist'''
        if exists(folder_path):
            try:
                shutil.rmtree(folder_path)
            except OSError as e:
                pass

        os.makedirs(folder_path)
        
        return folder_path

    def create_workSpace(self, video_path):
        self.video_name = basename(video_path).split('.')[0]

        self.resultSpace = self.create_folder(join(VIDEO_ROOT, self.video_name))

        self.cv_resultSpace = self.create_folder(join(self.resultSpace, 'CV_result'))
        self.id_faceFolder = self.create_folder(join(self.cv_resultSpace, 'ID_face'))

        self.asr_resultSpace = self.create_folder(join(self.resultSpace, 'ASR_result'))
        self.id_speakerFolder = self.create_folder(join(self.asr_resultSpace, 'ID_speaker'))

        

def main(video_path, dialog):
    
    # remove all speaker folder
    [ shutil.rmtree(i) for i in glob.glob('/mnt/c/Users/phudh/Desktop/src/dialog_system/Identify_speaker/speaker_id/*')]

    dialog.create_workSpace(video_path)
    
    # extrac audio from video
    audio_path = dialog.extrac_audio_from_video(video_path)

    # denoise audio
    audio_path = dialog.denoise_fullAudio(audio_path)

    # dialog.id_model.database.clean_database(None)
    dialog.inference(audio_path)

if __name__ == "__main__":
    # remove all speaker folder

    dialog = Dialog()
    

    # files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/dialog/dialog.wav'
    # main(files_path, dialog)
    # files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/baongam/baongam_denoise.wav'
    # main(files_path, dialog)
    # files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/hongnhan_tap21/hongnhan_tap21_denoise.wav'
    # main(files_path, dialog)
    # files_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/hồngnhan/hồngnhan_denoise.wav'
    # main(files_path, dialog)
    video_path = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/phim1.mp4'
    main(video_path, dialog)
    


    

    