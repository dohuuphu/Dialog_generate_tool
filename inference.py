
import shutil
import argparse
import sys
import os
from cv2 import imwrite
import torchaudio
import cv2 as cv
from os.path import dirname, join, basename, exists

sys.path.append('./STT')

from STT.speechbrain.pretrained import EncoderASR
from STT.asr_model.text_processing.inverse_normalize import InverseNormalizer
from STT.asr_model.audio import AudioFile
from STT.asr_model.variabels import *
from STT.asr_model.model import ASRModel


from Identify_speaker.inference import Model as ID_model
from Identify_speaker.variables import *
from Identify_speaker.uisrnn.inference import Diarization_model

from FaceRecognition.face_module import FaceRecognize, Facelandmark

from denoiser.denoiser.enhance_custom import Denoiser

from utils.variables import *

from moviepy.editor import *

import torch

import glob
import shutil
import numpy as np
import moviepy.editor as mp
from pydub import AudioSegment as am
from scipy.io import wavfile
from PIL import Image, ImageOps

class Dialog():
    def __init__(self) -> None:
        self.normalizer = InverseNormalizer("vi")
        self.asr_model = ASRModel()
        self.id_model = ID_model() 
        self.diarization = Diarization_model()
        self.denoiser = Denoiser()
        self.faceRecognize = FaceRecognize()
        self.faceLandmark = Facelandmark(MAX_NUM_FACES, STATIC_IMG_MODE, MIN_DETECTION_CONFIDENCE)
        self.video_name = None
        self.resultSpace = None
        self.cv_resultSpace = None
        self.id_faceFolder = None
        self.asr_resultSpace = None
        self.id_speakerFolder = None
        self.dialog_timeseries = []
        self.count_speaker = 0
        self.count_draf =0
        self.log = None

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

    def inference(self, audio_path, cross_check):
        self.count_speaker = 0
        speaker_ID =0
        list_text = []
        list_emb = []
        audio_ = AudioFile(audio_path)
        with open(audio_path.replace('.wav','.txt'), 'w') as w:
            for idx, (start, end, audio_buffer, s) in enumerate(audio_.split_file()):
                self.count_speaker += 1
                # speech to text
                audio_np = self.buffer_to_numpy(audio_, audio_buffer)

                # audio_np = self.denoise_numpyData(audio_np).astype(np.float64)
                audio_t = torch.from_numpy(audio_np).unsqueeze(1)
                text = self.asr_model.transcribe(audio_np)[0] if self.asr_model else "emtpy for debug only"
                list_text.append(text)
                
                list_emb.append(self.id_model.get_embedding(audio_np))

                # save wwav file for debug
                self.count_draf+=1
                path_audio = f'{DRAF_ASR}{self.count_draf}.wav'
                wavfile.write(path_audio, SAMPLE_RATE, audio_np.astype(np.float32) )

                
                # if len(list_emb) > 1:
                #     result = self.diarization.verify_speaker(list_emb)
                #     if 2 in result: # meet third speaker                       
                #         for idx, spk_id in enumerate(result):
                #             if spk_id != result[-1]:
                #                 info = f'{spk_id} - {list_text[idx]}\n'
                #                 w.writelines(info)
                #                 print(info)
                #         w.writelines('=========\n')
                #         print('=========\n')
                #         list_text = [list_text[-1]]
                #         list_emb = [list_emb[-1]]

                # save timeseries log
                self.dialog_timeseries.append([start/1000, end/1000, text])
                

            # Run for all audio
            result = self.diarization.verify_speaker(list_emb)
            for idx, spk_id in enumerate(result):
                info = f'{spk_id} - {list_text[idx]}\n'
                w.writelines(info)

    
    def is_sameSpeaker(self, method):
        '''Cross check by Computer Vision to determine the speaker in 2 subclips are the same person
        - Step 1: Trim video from time_segmentation that was save from audio_split
        - Step 2: Face detection + face verification => cluster the person in subclips (face + mouth)
        - Step 3: Using mouth motion to predict who was talking
        - Step 4: Return True if 2 person are the same, vice versa'''

        # trim sub_clip
        subClip_path = self.cut_person_inVideo()

        # cluster the person in subclips
        name_videos = []

        # clean folder
        self.id_faceFolder = self.create_folder(join(self.cv_resultSpace, 'ID_face')) 

        with open(join(self.cv_resultSpace, 'face_recognize.txt'), 'w') as wr:
            for subClip_path_ in glob.glob(f'{subClip_path}/*'):
                name_video = subClip_path_.split('/')[-1].split('.')[0]
                name_videos.append(name_video)

                # process sub_video
                try:
                    frame_index = 0
                    cap = cv.VideoCapture(subClip_path_)

                    if (cap.isOpened()== False):
                        print(f"Error opening {subClip_path_}")
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        frame_index += 1

                        if frame_index % SKIP_FRAME != 0: continue  #skip frame

                        if ret:
                            faces, mouths, _ = self.faceRecognize.get_face(frame)
                            faces_ = []
                            mouths_ = []

                            # detect_face again to get landmark at cutting_image
                            for idx, face in enumerate(faces):
                                img_pad = np.zeros([224, 224, 3])
                                img_pad[:face.shape[0], :face.shape[1], :] = face
                                # cv.imwrite(f'/mnt/c/Users/phudh/Desktop/src/dialog_system/draf_2/{name_video}_{frame_index+idx}_pad.jpg',img_pad)
                                face_, mouth_, landmark = self.faceRecognize.get_face(img_pad, crop_img = False, img_raw = face)

                                if len(face_) > 0:
                                    face_write  = face_[0].copy()
                                    for point in landmark[0]:
                                        cv.circle(face_write, (int(point[0]), int(point[1])), 3, (0,0,255), -1)
                                    cv.imwrite(f'{DRAF_CV}{name_video}_{frame_index+idx}.jpg',face_write)
                                    cv.imwrite(f'{DRAF_CV}{name_video}_{frame_index+idx}.jpg',face_[0])
                                    cv.imwrite(f'{DRAF_CV}{name_video}_{frame_index+idx}_mouth.jpg',mouth_[0])
                                    faces_.append(face_[0])
                                    mouths_.append(mouth_[0])

                            # cluster person
                            for idx, face in enumerate(faces):
                                stt, score, face_name_id, emb = self.faceRecognize.verify_face(face)

                                face_name = self.faceRecognize.database.map_storage[face_name_id] if len(self.faceRecognize.database.map_storage) > 0 else 'Null'
                                if stt is IDENTIFIED and len(self.faceRecognize.database.map_storage) > 0:
                                    info = f'face {face_name}'
                                    # if score >= 0.9:
                                    _ = self.save_face(self.id_faceFolder, emb, face, face_name, prefix = name_video, mouth=mouths[idx])
                                else: 
                                    info = self.save_face(self.id_faceFolder, emb, face, prefix = name_video, mouth=mouths[idx])

                                i = f'{info} | pred {face_name} : {score:.3f} \n'
                                wr.writelines(i)
                                print(i)
                        else:
                            break
                    
                except OSError as e:
                    # print(e)s
                    raise e
            if method == CHECK_BY_CV:
                return self.predict_sameSpeakerCV(name_videos) # predict 2 subclip are the same person or not
            elif method == CHECK_BY_LANDMARK:
                return self.predict_sameSpeakerLandmark(name_videos)

    def predict_sameSpeakerLandmark(self, name_videos):
        list_speaker = glob.glob(f'{self.id_faceFolder}/*')
        results = {}
        list_pred = []
        if len(list_speaker) == 1: return True

        #check mouth is a motion

        # loop in subclips
        for video_id in name_videos:
            # loop all speaker in a subclip
            for speaker_folder in list_speaker: 
                count_speech = 0
                frame_index = 0

                # loop all face of a speaker
                list_frame = []
                # get image by name and sorted
                for frame in glob.glob(f'{speaker_folder}/{video_id}*.jpg'):
                    if '_mouth' in frame: continue
                    name = frame.split('/')[-1].replace('.jpg','')
                    list_frame.append([int(name), frame])
                
                for _, frame in sorted(list_frame):

                    face = cv.imread(frame)
                    
                    # Get face landmark
                    face_landmark, face_mesh_results = self.faceLandmark.detectFacialLandmarks(face, self.faceLandmark.face_mesh_images, display=True)

                    # Check mouth is open or NOT
                    if face_mesh_results.multi_face_landmarks:
                        output_image, status = self.faceLandmark.isOpen(face, face_mesh_results, 'MOUTH', threshold=15, display=False, output_img=face_landmark)
                        cv.imwrite(f'{frame.replace(".jpg", "_landmark.jpg")}',output_image)# cv.hconcat([face, output_image]))
                    else: 
                        continue

                    frame_index += 1

                    if frame_index % 2 == 0:
                        frame_index -=1
                        
                        if status != first_stt:
                            count_speech +=1 
                            
                    first_stt = status
            
                results.update({speaker_folder[-1]: count_speech})
                print(f'video {video_id} | speaker_folder {speaker_folder[-1]}: {count_speech}')

            if self.is_dictNotZero(results):
                list_maxValue = []
                ranking = sorted(results.items(), key=lambda x: x[1])
                max_value = max(ranking, key=lambda x: x[1])
                for val in ranking:
                    if val[1] == max_value[1]:
                        list_maxValue.append(val[0])

                list_pred.append(list_maxValue) # sort result by count_speech and return speaker_name with max count_speech
                print(f'max_speech : {list_pred[-1]}')
            else:
                list_pred.append([''])

            # #clean folder 
            # self.faceRecognize.database.clean_database()
            # self.cv_resultSpace = self.create_folder(join(self.resultSpace, 'CV_result'))
            # self.id_faceFolder = self.create_folder(join(self.cv_resultSpace, 'ID_face'))

        if '' in list_pred[0] and '' in list_pred[1]:
            return False
        for val in list_pred[1]:
            if val in list_pred[0]:
                return True
        return False
           
    def predict_sameSpeakerCV(self, name_videos):
        list_speaker = glob.glob(f'{self.id_faceFolder}/*')
        results = {}
        list_pred = []
        if len(list_speaker) == 1: return True

        #check mouth is a motion

        # loop in subclips
        for video_id in name_videos:
            # loop all speaker in a subclip
            for speaker_folder in list_speaker: 
                count_speech = 0
                frame_index = 0
                # loop all mouth_img of a speaker
                for frame in glob.glob(f'{speaker_folder}/{video_id}*_mouth.jpg'):

                    mouth = cv.imread(frame)
                    frame_index += 1

                    gray_mouth = cv.cvtColor(mouth, cv.COLOR_BGR2GRAY)
                    gray_mouth = cv.GaussianBlur(gray_mouth, (3, 3), 0)

                    if frame_index % 2 == 0:
                        frame_index = 0
                        des_width = max(fist_mouth.shape[0], gray_mouth.shape[0])
                        des_height = max(fist_mouth.shape[1], gray_mouth.shape[1])


                        fist_mouth_ = cv.resize(fist_mouth, (des_height, des_width), interpolation = cv.INTER_AREA)
                        gray_mouth_ = cv.resize(gray_mouth, (des_height, des_width), interpolation = cv.INTER_AREA)
                        # cv.imwrite(f'/mnt/c/Users/phudh/Desktop/src/dialog_system/draf_2/fist_mouth_{frame.split("/")[-1]}', fist_mouth)
                        # cv.imwrite(f'/mnt/c/Users/phudh/Desktop/src/dialog_system/draf_2/gray_mouth_{frame.split("/")[-1]}', gray_mouth)
                        # Compare the two frames, find the difference
                        frame_delta = cv.absdiff(fist_mouth_, gray_mouth_)
                        # cv.imwrite(f'/mnt/c/Users/phudh/Desktop/src/dialog_system/draf_2/frame_delta_{frame.split("/")[-1]}',frame_delta)
                        thresh = cv.threshold(frame_delta, 25, 255, cv.THRESH_BINARY)[1]

                        # Fill in holes via dilate(), and find contours of the thesholds
                        thresh = cv.dilate(thresh, None, iterations = 2).astype(np.uint8)
                        # cv.imwrite(f'/mnt/c/Users/phudh/Desktop/src/dialog_system/draf_2/thresh_move_{frame.split("/")[-1]}',thresh)
                        cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                        
                        # loop over the contours
                        for c in cnts:

                            # Save the coordinates of all found contours
                            (x, y, w, h) = cv.boundingRect(c)
                            
                            # movement
                            if cv.contourArea(c) > MOTION_THRESHOLD:
                                # Draw a rectangle around big enough movements
                                cv.rectangle(mouth, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                # cv.imwrite('/mnt/c/Users/phudh/Desktop/src/dialog_system/draf_2/move.jpg',mouth)
                                count_speech+=1
                                break
                            
                    fist_mouth = gray_mouth
            
                results.update({speaker_folder[-1]: count_speech})
                print(f'speaker_folder {speaker_folder[-1]}: {count_speech}')

            if self.is_dictNotZero(results):
                list_pred.append(sorted(results.items(), key=lambda x: x[1])[-1][0]) # sort result by count_speech and return speaker_name with max count_speech
                print(f'max_speech : {list_pred[-1]}')
            else:
                list_pred.append('')

            # #clean folder 
            # self.faceRecognize.database.clean_database()
            # self.cv_resultSpace = self.create_folder(join(self.resultSpace, 'CV_result'))
            # self.id_faceFolder = self.create_folder(join(self.cv_resultSpace, 'ID_face'))
        
        return True if list_pred[-2] == list_pred[1] else False

    def cut_person_inVideo(self):
        ''' Cut sub_clip depending on audio segment'''
        result_folder = join(self.cv_resultSpace, 'face_video')
        self.create_folder(result_folder)
        # video_path = join(VIDEO_ROOT, f'{self.video_name}.mp4')
        try:
            video = VideoFileClip(video_path)
            for idx, sentence in enumerate(self.dialog_timeseries[-2:]):
                start_time, end_time, _ = sentence
                cutting = video.subclip(start_time, end_time)
                cutting.write_videofile(f"{join(result_folder, str(idx))}.mp4")
                
            return result_folder

        except OSError as e:
            raise e

    def is_dictNotZero(self, dict):
        ''' Check dict is not a Zero_dict'''
        for value in dict.values():
            if value != 0:
                return True
        return False

    def save_speaker(self, root_folder, emb, audio, name = None):
        ''' save embedded feature of speaker'''   
        if name is None or name == "Null":
            self.count_speaker+=1
            name = str(self.count_speaker)

        success, emb_path = self.id_model.save_newEmb( root_folder, emb = emb, name = name)

        # save audio with emb
        wavfile.write(emb_path.replace('.txt', '.wav'), SAMPLE_RATE, audio.astype(np.float32) )

        return f'New_speaker *{self.count_speaker}*' if success else f'Save new speaker failed!!!'

    def save_face(self, root_folder, emb, face, name = None, prefix = '', mouth = None):
        ''' save embedded feature of face'''   

        success, emb_path =  self.faceRecognize.save_newEmb(root_folder, name, emb, prefix)
        
        cv.imwrite(emb_path.replace('.txt', '.jpg'), face)

        if mouth is not None: 
            cv.imwrite(emb_path.replace('.txt', '_mouth.jpg'), mouth)

        return f'New_face *{self.faceRecognize.database.num_speaker}*' if success else f'Save new speaker failed!!!'

    def extract_audio_from_video(self, video_path):
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

    def create_folder(self, folder_path, remain = False):
        ''' create folder with path, remove old folder if it exist'''
        if exists(folder_path):
            try:
                if remain: return folder_path
                shutil.rmtree(folder_path)
            except OSError as e:
                pass

        os.makedirs(folder_path)
        
        return folder_path

    def create_workSpace(self, video_path):
        self.video_name = basename(video_path).split('.')[0]

        self.create_folder(VIDEO_ROOT, remain=True)
        self.create_folder(DRAF_ASR)
        self.create_folder(DRAF_CV)

        self.resultSpace = self.create_folder(join(VIDEO_ROOT, self.video_name))

        self.cv_resultSpace = self.create_folder(join(self.resultSpace, 'CV_result'))
        self.id_faceFolder = self.create_folder(join(self.cv_resultSpace, 'ID_face'))

        self.asr_resultSpace = self.create_folder(join(self.resultSpace, 'ASR_result'))
        self.id_speakerFolder = self.create_folder(join(self.asr_resultSpace, 'ID_speaker'))

        self.log = open(f'{self.resultSpace}/log.txt', 'w') 

        

def main(arg, dialog):
    video_path = arg if type(arg) is str else arg.path
    dialog.create_workSpace(video_path)
    
    # extract audio from video
    audio_path = dialog.extract_audio_from_video(video_path)

    # denoise audio
    audio_path = dialog.denoise_fullAudio(audio_path)

    dialog.inference(audio_path, False)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--path', type=str, help='path of video', required=True)
    # parser.add_argument('-c', '--cross_check', type=bool, help='cross_check same person by Computer Vision', default=True)
    # args = parser.parse_args()

    dialog = Dialog()
    
    args = '/mnt/c/Users/phudh/Desktop/src/dialog_system/video/baongam.mp4'
    main(args, dialog)
    


    

    