import os
import subprocess
import tqdm
import datetime
from asr_model.audio import AudioFile
from asr_model.utils import extract_audio, convert_audio, write_to_file
from pydub import AudioSegment as am
from .utils import *
import wave
from asgiref.sync import async_to_sync
import threading
from queue import  Queue
import time
from webrtcvad import Vad

class RealtimeGenerator:
    exit_event = threading.Event()
    def __init__(self, 
        asr_model,
        normalizer,
        gector=None,
        bytes_data='',
        channel_id=None,
        src_lang='vi', 
        split_threshold=200,
        split_duration=5000, 
        max_words=12, ):
        super(RealtimeGenerator, self).__init__()

        self.channel_id = channel_id
        self.max_words = max_words
        self.split_duration = split_duration
        self.split_threshold = split_threshold
        self.src_lang = src_lang
        self.model = asr_model
        self.itn = normalizer
        self.gector = gector
        self.bytes_data = bytes_data

        if gector is not None:
            self.max_len = gector.max_len * 2
        else:
            self.max_len = 32


    # def stop(self):
    #     """stop the asr process"""
    #     RealtimeGenerator.exit_event.set()
    #     self.asr_input_queue.put("close")
    #     print("asr stopped")
 
    def start(self):
        """start the asr process"""
        self.last_buffer = b''
        self.final_text = Queue()
        self.transcript = threading.Thread(target=RealtimeGenerator.transcript, args=(
            self, self.bytes_data,))
        self.transcript.start()

        self.get_final = threading.Thread(target=RealtimeGenerator.get_final, args=(
            self,))
        self.get_final.start()

    
    def post_process(self, tokens):
        final_tokens = self.itn.inverse_normalize(tokens, verbose=False)
        if self.gector:
            final_batch = self.gector.handle_batch([final_tokens.split(' ')])
            final_tokens = final_batch[0][0]
        final_transcript = final_tokens

        return final_transcript, final_tokens

    def transcript(self,bytes_data):
        aggressiveness = 3
        self.remain_buffer= b''
        frames=b''
        segment_buffer=b''
        triggered = False
        ring_buffer = collections.deque(maxlen=3)
        voiced_frames = []
        frame_duration_ms = 30
        frame_index = 0
        threshold=0.5
        while True: 
            vad = Vad(int(aggressiveness))
            input_buffer = bytes_data.get()
            
            if len(input_buffer) > 960:
                if len(self.remain_buffer)>0:
                    frames = self.remain_buffer + input_buffer[0:(960 - len(self.remain_buffer))]
                    if 960 - len(self.remain_buffer) == 0:
                        self.remain_buffer = b''
                    else:
                        self.remain_buffer = input_buffer[(960 - len(self.remain_buffer)):]
                else:
                    self.remain_buffer = input_buffer[960:]
                    frames = input_buffer[0:960]
                    
            is_speech =vad.is_speech(frames,16000)
            if not triggered:
                ring_buffer.append((frames, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > threshold * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer :
                        if s:
                            voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frames)
                ring_buffer.append((frames, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > threshold * ring_buffer.maxlen:
                    triggered = False
                    segment_buffer = b''.join(voiced_frames)
                    samples = pcm_to_np(segment_buffer, DEFAULT_FORMAT)
                    audio = np.squeeze(samples)
                    start_time = time.time()
                    tokens = self.model.transcribe(audio)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    
                    self.final_text.put(tokens)
                    ring_buffer.clear()
                    voiced_frames = []
                    

            
            # if len(tokens) > 1:
            #     self.last_buffer = b''
            frame_index+=1

    def get_final(self):
        while True: 
            final_text = self.final_text.get()
            print('token',final_text)


        # for start, end, audio in self.audio_file.split():


        #     if end_split == True and count_split != 1:
        #         pre= start
        #     tokens = self.model.transcribe(audio)[0]
        #     if trans_dict is not None:
        #         if start - trans_dict.get('end', 0) > self.split_threshold or len(trans_dict['tokens']) > self.max_len:
        #             final_transcript, final_tokens = self.post_process(trans_dict['tokens'])
        #             results.append([pre,last,final_transcript.strip()])
        #             trans_dict = None
        #             end_split = True
        #         else:
        #             trans_dict['tokens']+=' '+tokens
        #             trans_dict['end'] = end
        #             end_split = False
            
        #     if trans_dict is None:
        #         trans_dict = {
        #             'tokens': tokens,
        #             'start': start,
        #             'end': end,
        #         }
            
        #     last = end
        #     count_split+=1

        # if trans_dict is not None:
        #     final_transcript, final_tokens = self.post_process(trans_dict['tokens'])
        #     results.append([pre,last,final_transcript.strip()])
        # async_to_sync(layer.group_send)(
        #         self.channel_id, {
        #         'type': 'send_message',
        #         'message': '100'
        # })
        # self.audio_file.close()
        # self.close_file()

        # return [{'time':str(datetime.timedelta(seconds=round(int(result[0]/1000))))+' - '+str(datetime.timedelta(seconds=round(int(result[1]/1000)))),'transcript': result[2]} for result in results  ]