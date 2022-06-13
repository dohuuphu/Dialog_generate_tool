
import collections
import time
from webrtcvad import Vad
import numpy as np
from utils.utils import pcm_to_np
from variables import *

def consumers(socketio, verify_model):
    
    vad = Vad(int(3))
    threshold=0.6
    temp_buffer={}

    
    @socketio.on('connect')
    async def connect(sid,environ):
        print("CONNECTED: " + sid)
        temp_buffer[sid] = {'remain_buffer':b'','triggered':False,'ring_buffer':collections.deque(maxlen=10),'voiced_frames':[],'line_sent':0}
        print("calling user: ",len(temp_buffer))

    @socketio.on('disconnect')
    async def disconnect(sid):
        print("DISCONNECTED: " + sid)
        del temp_buffer[sid]
        print("calling user: ",len(temp_buffer))



    @socketio.on('buffer_to_server')
    async def buffer_to_server(sid,data):
            
        input_buffer =(data['buffer'])
        type_buffer = (data['type'])
        

        remain_buffer= temp_buffer[sid]['remain_buffer']
        triggered = temp_buffer[sid]['triggered']
        ring_buffer = temp_buffer[sid]['ring_buffer']
        voiced_frames = temp_buffer[sid]['voiced_frames']

        frames=b''
        segment_buffer=b''
        
        if len(remain_buffer) > 0:
            input_buffer = remain_buffer + input_buffer
            temp_buffer[sid]['remain_buffer'] = b''
                
        if len(input_buffer) > 960:
            num_loop_buffer = len(input_buffer)//960
            if len(input_buffer)%960 != 0:
                temp_buffer[sid]['remain_buffer'] = input_buffer[960*num_loop_buffer:]

            for i in range(0,num_loop_buffer):
                
                triggered = temp_buffer[sid]['triggered']
                frames = input_buffer[960*i:960*(i+1)]
                is_speech =vad.is_speech(frames,16000)
                    
                if not triggered:
                    ring_buffer.append((frames, is_speech))
                    temp_buffer[sid]['ring_buffer'] = ring_buffer
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    if num_voiced > threshold * ring_buffer.maxlen:
                        temp_buffer[sid]['triggered'] = True
                        for f, s in ring_buffer :
                            if s:
                                voiced_frames.append(f)
                                temp_buffer[sid]['voiced_frames'] = voiced_frames
                        ring_buffer.clear()
                        temp_buffer[sid]['ring_buffer'] = collections.deque(maxlen=10)
                else:
                    voiced_frames.append(frames)
                    temp_buffer[sid]['voiced_frames'] = voiced_frames
                    ring_buffer.append((frames, is_speech))
                    temp_buffer[sid]['ring_buffer'] = ring_buffer
                    num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                    if num_unvoiced > threshold * ring_buffer.maxlen:
                        temp_buffer[sid]['triggered'] = False
                        segment_buffer = b''.join(voiced_frames)
                        samples = pcm_to_np(segment_buffer, DEFAULT_FORMAT)
                        audio = np.squeeze(samples)
                        result, score, speaker = verify_model.verify_speakers_buffer(audio)
                        await socketio.emit('buffer_to_client', f'Status: {result}, score: {score}, speaker: {speaker}',room=sid)
                        
                        
                        ring_buffer.clear()
                        temp_buffer[sid]['ring_buffer'] = collections.deque(maxlen=10)
                        voiced_frames = []
                        temp_buffer[sid]['voiced_frames'] = []