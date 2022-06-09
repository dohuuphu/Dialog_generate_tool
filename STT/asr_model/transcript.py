import os
import subprocess
import tqdm
import datetime
from asr_model.audio import AudioFile
from asr_model.utils import extract_audio, convert_audio, write_to_file
from pydub import AudioSegment as am
from channels.layers import get_channel_layer
import time
from asgiref.sync import async_to_sync
import glob
from pydub import AudioSegment
import os
import pathlib
import torch
import torchaudio

VIDEO_EXT = ['mp4', 'ogg', 'm4v', 'm4a', 'webm', 'flv', 'amv', 'avi']
AUDIO_EXT = ['mp3', 'flac', 'wav', 'aac', 'm4a', 'weba', 'sdt']


class SubGenerator:
    def __init__(self, 
        file_path, 
        asr_model,
        normalizer,
        gector=None,
        channel_id=None,
        src_lang='vi', 
        split_threshold=200,
        split_duration=5000, 
        max_words=12, 
        sub_format=['srt'], 
        output_directory="./temp"):
        super(SubGenerator, self).__init__()

        if not os.path.exists(file_path):
            raise ValueError("File does not exist: %s" % file_path)

        self.file_path = file_path
        self.channel_id = channel_id
        self.file_name, self.file_ext = os.path.split(file_path)[-1].split(".")
        self.max_words = max_words
        self.temp_path = os.path.join("temp", self.file_name + ".wav")
        self.split_duration = split_duration
        self.split_threshold = split_threshold
        self.src_lang = src_lang
        self.model = asr_model
        self.itn = normalizer
        self.gector = gector
        if gector is not None:
            self.max_len = 0
        else:
            self.max_len = 0

        if not os.path.exists("temp"):
            os.makedirs("temp")

        if self.file_ext in VIDEO_EXT:
            self.is_video = True
            extract_audio(self.file_path, self.temp_path)
        elif self.file_ext in AUDIO_EXT:
            self.is_video = False
            convert_audio(self.file_path, self.temp_path)
        else:
            raise ValueError("Extension mismatch")

        self.sub_format = sub_format
        self.output_directory = output_directory
        self.output_file_handle_dict = {}
    
    def post_process(self, tokens):
        final_tokens = self.itn.inverse_normalize(tokens, verbose=False)
        if self.gector:
            final_batch = self.gector.handle_batch([final_tokens.split(' ')])
            final_tokens = final_batch[0][0]
        final_transcript = final_tokens

        return final_transcript, final_tokens

    def create_sub(self):
        self.audio_file = AudioFile(self.temp_path)
        for format in self.sub_format:
            output_filename = os.path.join(
                self.output_directory, self.file_name + "." + format)
            print("Creating file: " + output_filename)
            self.output_file_handle_dict[format] = open(
                output_filename, mode="w", encoding="utf-8")
            # For VTT format, write header
            if format == "vtt":
                self.output_file_handle_dict[format].write("WEBVTT\n")
                self.output_file_handle_dict[format].write("Kind: captions\n\n")
        progress_bar = tqdm.tqdm(
            total=int(self.audio_file.audio_length * 1000))
        line_count = 1
        last = 0
        trans_dict = None
        for start, end, audio in self.audio_file.split():
            _, tokens, _ = self.model.transcribe_with_metadata(audio, start)[0]
            if trans_dict is not None:
                if start - trans_dict.get('end', 0) > self.split_threshold or len(trans_dict['tokens']) > self.max_len:
                    final_transcript, final_tokens = self.post_process(trans_dict['tokens'])
                    line_count = self.write_sub(
                        final_transcript, final_tokens, trans_dict['start'], 
                        trans_dict['end'], line_count, trans_dict['split_points']
                    )
                    trans_dict = None
                else:
                    trans_dict['tokens'].extend(tokens)
                    trans_dict['split_points'].append(trans_dict['end'])
                    trans_dict['end'] = end
            
            if trans_dict is None:
                trans_dict = {
                    'tokens': tokens,
                    'start': start,
                    'end': end,
                    'split_points': [],
                }

            progress_bar.update(int(end - last))
            last = end
        
        if trans_dict is not None:
            final_transcript, final_tokens = self.post_process(trans_dict['tokens'])
            line_count = self.write_sub(
                            final_transcript, final_tokens, trans_dict['start'], 
                            trans_dict['end'], line_count, trans_dict['split_points']
                        )

        self.audio_file.close()
        self.close_file()

        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
    
    def write_sub(self, transcript, tokens, start, end, line_count, split_points=[]):
        if split_points is None:
            split_points = []
        split_points.append(1e8)

        if end - start > self.split_duration:
            infer_text = ""
            num_inferred = 0
            split_idx = 0
            prev_start = start

            for token in tokens:
                if (
                    num_inferred > self.max_words 
                    or token['start'] > split_points[split_idx] 
                    or token['start'] > prev_start + self.split_duration
                ):
                    write_to_file(self.output_file_handle_dict, infer_text,
                                  line_count, (prev_start / 1000, prev_end / 1000))
                    line_count += 1
                    infer_text = ""
                    num_inferred = 0
                    prev_start = token['start']

                    if token['start'] > split_points[split_idx]:
                        split_idx += 1

                infer_text += token['text'] + " "
                num_inferred += 1
                prev_end = token['end']

            if infer_text:
                write_to_file(self.output_file_handle_dict, infer_text,
                                line_count, (prev_start / 1000, token['end'] / 1000))
                line_count += 1
        else:
            write_to_file(self.output_file_handle_dict, transcript,
                            line_count, (start / 1000, end / 1000))
            line_count += 1
        
        return line_count

    def sync_sub(self):
        if "srt" not in self.sub_format:
            return
        srt_path = os.path.join(self.output_directory, self.file_name + ".srt")
        sync_path = os.path.join(
            self.output_directory, self.file_name + "_synchronized.srt")
        cmd = f"ffsubsync {self.file_path} -i {srt_path} -o {sync_path}"

        subprocess.call(cmd, shell=True)

        if os.path.exists(sync_path):
            os.remove(srt_path)
            os.rename(sync_path, srt_path)

    def close_file(self):
        for format in self.output_file_handle_dict.keys():
            self.output_file_handle_dict[format].close()

    def add_sub_to_video(self):
        if self.is_video:
            srt_path = os.path.join(
                self.output_directory, self.file_name + ".srt")
            out_path = os.path.join(
                self.output_directory, self.file_name + "_sub.mp4")
            cmd = f"ffmpeg -loglevel quiet -i {self.file_path} -i {srt_path} -y -c copy -c:s mov_text {out_path}"
            os.system(cmd)
    
    def transcript_split(self):
        print('tempppppppp', self.temp_path)
        sample_rate = 16000
        sound = am.from_file(self.temp_path, format='wav')
        sound = sound.set_frame_rate(sample_rate)
        sound = sound.set_channels(1)
        sound.export(self.temp_path, format='wav')
        self.audio_file = AudioFile(self.temp_path)
        total=int(self.audio_file.audio_length * 1000)
        progress_bar = tqdm.tqdm(
            total=total)
        last = 0
        trans_dict = None
        end_split = None
        count_split = 0
        last_index=0
        last_s=''
        path = '/mnt/c/Users/phudh/Documents/source/label_TTS/data_genarate/'
        num_speaker = len(glob.glob(path+'*/'))+1
        speaker_path = path+'speaker_'+str(num_speaker)
        pathlib.Path(speaker_path).mkdir()
        index = len(glob.glob(speaker_path+"/*.wav"))+1
        for start, end, audio, s in self.audio_file.split_file():
            if trans_dict is not None:
                if start - trans_dict.get('end', 0) > self.split_threshold or len(trans_dict['tokens']) > self.max_len:
                    final_transcript = trans_dict['tokens']
                    
                    if len(final_transcript)>0 and final_transcript!= ' ': 
                        with open(speaker_path+".txt", "a") as myfile:
                            myfile.write("audio/"+str(last_index)+'.wav|'+final_transcript.strip()+'|'+str(num_speaker)+'\n')
                        with open(speaker_path+"/"+str(last_index)+".txt", "a") as myfile1:
                            myfile1.write(final_transcript.strip())
                        index+=1
                        
                    trans_dict = None
                    end_split = True
            temppath = speaker_path+'/'+str(index)+'.wav'
            AudioSegment.from_raw(file=s, sample_width=2, frame_rate=16000, channels=1).export(temppath, format='wav')
            signal, sr = torchaudio.load(temppath, channels_first=False)
            tokens = self.model.transcribe_batch(self.model.audio_normalizer(signal,sr).unsqueeze(0),torch.tensor([1.0]))[0]
            if trans_dict is None:
                trans_dict = {
                    'tokens': tokens,
                    'start': start,
                    'end': end,
                } 
                
    
            progress_bar.update(int(end - last))
            last = end
            count_split+=1
            last_index = index
            last_s=s
            

        if trans_dict is not None:
            final_transcript = trans_dict['tokens']
            if len(final_transcript.strip())>0:
                with open(speaker_path+".txt", "a") as myfile:
                    myfile.write("audio/"+str(last_index+1)+'.wav|'+final_transcript.strip()+'|'+str(num_speaker)+'\n')
                with open(speaker_path+"/"+str(last_index+1)+".txt", "a") as myfile1:
                    myfile1.write(final_transcript.strip())
        self.audio_file.close()
        self.close_file()

        return 0