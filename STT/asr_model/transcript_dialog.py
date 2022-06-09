import os
from asr_model.audio import AudioFile
from asr_model.utils import extract_audio, convert_audio, write_to_file
from pydub import AudioSegment as am
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import os
import torch
import torchaudio

from asr_model.variabels import *

VIDEO_EXT = ['mp4', 'ogg', 'm4v', 'm4a', 'webm', 'flv', 'amv', 'avi']
AUDIO_EXT = ['mp3', 'flac', 'wav', 'aac', 'm4a', 'weba', 'sdt']


class SubGenerator:
    def __init__(self, 
        asr_model,
        normalizer,
        gector=None,
        output_directory="./temp"):
        super(SubGenerator, self).__init__()

        self.max_words = 12
        self.split_duration = 5000
        self.split_threshold = 200
        self.model = asr_model
        self.itn = normalizer
        self.gector = gector
        if gector is not None:
            self.max_len = 0
        else:
            self.max_len = 0

        if not os.path.exists("temp"):
            os.makedirs("temp")

        # if self.file_ext in VIDEO_EXT:
        #     self.is_video = True
        #     extract_audio(self.file_path, self.temp_path)
        # elif self.file_ext in AUDIO_EXT:
        #     self.is_video = False
        #     convert_audio(self.file_path, self.temp_path)
        # else:
        #     raise ValueError("Extension mismatch")

        self.output_directory = output_directory
        self.output_file_handle_dict = {}
    
    
    def format_audio(self, audio_path):
        sound = am.from_file(audio_path, format='wav')
        sound = sound.set_frame_rate(SAMPLE_RATE)
        sound = sound.set_channels(1)
        sound.export(audio_path, format='wav')
        return sound


    def transcript_split(self, audio_path):
        sound = self.format_audio(audio_path)
        
        

        last = 0
        trans_dict = None
        end_split = None
        count_split = 0
        last_index=0
        last_s=''
        

        for start, end, audio, s in audio_file.split_file():
            if trans_dict is not None:
                if start - trans_dict.get('end', 0) > self.split_threshold or len(trans_dict['tokens']) > self.max_len:
                    final_transcript = trans_dict['tokens']
                    
                    # if len(final_transcript)>0 and final_transcript!= ' ': 
                    #     with open(speaker_path+".txt", "a") as myfile:
                    #         myfile.write("audio/"+str(last_index)+'.wav|'+final_transcript.strip()+'|'+str(num_speaker)+'\n')
                    #     with open(speaker_path+"/"+str(last_index)+".txt", "a") as myfile1:
                    #         myfile1.write(final_transcript.strip())
                    #     index+=1
                        
                    trans_dict = None
                    end_split = True
            # temppath = speaker_path+'/'+str(index)+'.wav'
            # AudioSegment.from_raw(file=s, sample_width=2, frame_rate=16000, channels=1).export(temppath, format='wav')
            signal, sr = torchaudio.load('./a.wav', channels_first=False)
            tokens = self.model.transcribe(self.model.audio_normalizer(signal, SAMPLE_RATE).unsqueeze(0),torch.tensor([1.0]))[0]
            if trans_dict is None:
                trans_dict = {
                    'tokens': tokens,
                    'start': start,
                    'end': end,
                } 
                
            print('tokens ', tokens)
            last = end
            count_split+=1
            # last_index = index
            last_s=s
            

        # if trans_dict is not None:
        #     final_transcript = trans_dict['tokens']
        #     if len(final_transcript.strip())>0:
        #         with open(speaker_path+".txt", "a") as myfile:
        #             myfile.write("audio/"+str(last_index+1)+'.wav|'+final_transcript.strip()+'|'+str(num_speaker)+'\n')
        #         with open(speaker_path+"/"+str(last_index+1)+".txt", "a") as myfile1:
        #             myfile1.write(final_transcript.strip())


        return 0