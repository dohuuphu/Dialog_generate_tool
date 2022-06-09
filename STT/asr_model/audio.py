from .utils import *
import wave
import io

class AudioFile:
    def __init__(self, audio_path, audio_format=DEFAULT_FORMAT):
        super(AudioFile, self).__init__()

        self.audio_path = audio_path

        self.wav_file = wave.open(audio_path)
        self.audio_length = self.wav_file.getnframes() / self.wav_file.getframerate()
        self.audio_format = read_audio_format_from_wav_file(self.wav_file)
        # assert self.audio_format == audio_format, "Audio format mismatch"
        # print(self.audio_path)
        # assert ".ogg" in self.audio_path, "Audio format mismatch"

    def split(self, aggressiveness=3):
        
        frames = read_frames_from_file(self.wav_file)
        
        segments = vad_split(frames, aggressiveness=aggressiveness)
        
        for segment in segments:
            
            segment_buffer, time_start, time_end = segment  
            samples = pcm_to_np(segment_buffer, self.audio_format)
            yield time_start, time_end, np.squeeze(samples)
    
    def split_file(self, aggressiveness=3):
            
        frames = read_frames_from_file(self.wav_file)
        segments = vad_split(frames, aggressiveness=aggressiveness)
        for segment in segments:
            segment_buffer, time_start, time_end = segment  
            samples = pcm_to_np(segment_buffer, self.audio_format)
            s = io.BytesIO(segment_buffer)
            yield time_start, time_end, segment_buffer,s



    def close(self):
        self.wav_file.close()
