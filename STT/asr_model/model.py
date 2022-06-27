from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel, build_ctcdecoder
import torch
import numpy as np
import time
import tqdm
from multiprocessing import Pool
import warnings
from pydub import AudioSegment as am
from .audio import AudioFile
import datetime
from config.config import  get_config
import time
WINDOW_SIZE = 25
STRIDES = 20

# warnings.filterwarnings("error")

config_app = get_config()
class ASRModel:
    _instance = None
    _isInit = False
    def __init__(cls, cfg=None, *args, **kwargs):
        if not cls._isInit:
            cls.model_path = "STT/asr_model/base"
            cls.lm_path = "STT/asr_model/base/lm.bin"
            cls.hot_words = []

            print("Loading model...")
            start = time.time()
            cls.device = torch.device(
                config_app['deployment']['device'] if torch.cuda.is_available() else "cpu")
            cls.processor = Wav2Vec2Processor.from_pretrained("STT/asr_model/base")
            cls.model = Wav2Vec2ForCTC.from_pretrained("STT/asr_model/base").to(cls.device)
            cls.model.gradient_checkpointing_enable()
            print("Model loaded successfully in %fs" % (time.time() - start))

            # Sanity check
            x = torch.zeros([1, 10000]).to(cls.device)
            with torch.no_grad():
                out = cls.model(x).logits
                cls.vocab_size = out.shape[-1]

            cls.decoder = cls.build_lm(cls.processor.tokenizer)
            print("Language model loaded successfully in %fs" %
                (time.time() - start))   
            cls._isInit = True
    def build_lm(cls, tokenizer):
        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key)
                            for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:cls.vocab_size]
        vocab_list = vocab
        vocab_list[tokenizer.pad_token_id] = ""

        vocab_list[tokenizer.word_delimiter_token_id] = " "

        decoder = build_ctcdecoder(vocab_list, cls.lm_path)
        return decoder

    

    def transcribe_file(cls, audio_path):
        sound = am.from_file(audio_path, format='wav')
        sound = sound.set_frame_rate(16000)
        sound.export(audio_path, format='wav')
        audio_file = AudioFile(audio_path)
        progress_bar = tqdm.tqdm(total=int(audio_file.audio_length * 1000))
        last = 0
        result = []
        for start, end, audio in audio_file.split():
            transcript = ""
            try:
                transcript = cls.transcribe(audio)[0]
            except Exception as e: pass

            result.append((start, end, transcript.strip()))

            progress_bar.update(int(end - last))
            last = end

        audio_file.close()
        
        return [{'time': str(datetime.timedelta(seconds=round(start/1000)))+' - '+str(datetime.timedelta(seconds=round(end/1000))),
                 'transcript': transcript} for start, end, transcript in result]

    def transcribe_file_with_metadata(cls, audio_path):
        audio_file = AudioFile(audio_path)
        progress_bar = tqdm.tqdm(total=int(audio_file.audio_length * 1000))
        last = 0
        result = []
        for start, end, audio in audio_file.split():
            transcript, tokens, score = cls.transcribe_with_metadata(audio, start)[
                0]
            result.append((start, end, transcript.strip(), tokens, score))

            progress_bar.update(int(end - last))
            last = end
        
        audio_file.close()

        return [{'start': start,
                 'end': end,
                 'transcript': transcript,
                 'tokens': tokens,
                 'score': score} for start, end, transcript, tokens, score in result]

    def transcribe(cls, audio):
        if len(audio.shape) == 1:
            audio = audio[np.newaxis, :]
        elif len(audio.shape) > 2:
            raise ValueError(
                "Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))
        inputs = cls.processor(
            [x for x in audio], sampling_rate=16_000, return_tensors="pt", padding=True).to(cls.device)
        with torch.no_grad():
            attention_mask = None if not hasattr(
                inputs, 'attention_mask') else inputs.attention_mask
            logits = cls.model(inputs.input_values,
                                attention_mask=attention_mask,
                                ).logits
            pred_str = [cls.decoder.decode(
              logit.detach().cpu().numpy(), beam_width=100) for logit in logits]
        
        # predicted_ids = torch.argmax(logits, dim=-1)        
        # transcription = cls.processor.batch_decode(predicted_ids)[0]

        # print(ground_truth, pred_str)

        return pred_str

    def transcribe_with_metadata(cls, audio, start):
        if len(audio.shape) == 1:
            audio = audio[np.newaxis, :]
        elif len(audio.shape) > 2:
            raise ValueError(
                "Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))

        inputs = cls.processor(
            [x for x in audio], sampling_rate=16_000, return_tensors="pt", padding=True).to(cls.device)
        with torch.no_grad():
            attention_mask = None if not hasattr(
                inputs, 'attention_mask') else inputs.attention_mask
            logits = cls.model(inputs.input_values,
                                attention_mask=attention_mask,
                                ).logits

        with Pool() as pool:
            beam_batch = cls.decoder.decode_beams_batch(
                pool=pool, logits_list=logits.cpu().detach().numpy(), beam_width=500)

        pred_batch = []
        for top_beam in beam_batch:
            beam = top_beam[0]
            tokens = []
            score = beam[3]

            for w, i in beam[1]:
                tokens.append({
                    'text': w,
                    'start': start + i[0] * STRIDES,
                    'end': start + i[1] * STRIDES + WINDOW_SIZE,
                })

            pred_batch.append((beam[0], tokens, score))

        return pred_batch
