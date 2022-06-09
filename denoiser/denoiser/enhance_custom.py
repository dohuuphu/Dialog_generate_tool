from charset_normalizer import logging
import torch
import os
import numpy as np
import logging
import torchaudio

from concurrent.futures import ProcessPoolExecutor

from . import distrib, pretrained
from .audio import Audioset, find_audio_files
from .utils import LogProgress

logger = logging.getLogger(__name__)

class Denoiser():
    def __init__(self, alpha = 0) -> None:
        self.device = 'cpu'
        self.num_workers = 10
        self.model = pretrained.dns64().to(self.device).eval()
        self.alpha = alpha
        

    def enhance_numpyData(self, noisy_audio):
        noisy_audio_ = torch.from_numpy(noisy_audio.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            estimate = self.model(noisy_audio_)
            wav = (1 - self.alpha) * estimate + self.alpha * noisy_audio_
            wav = estimate / max(wav.abs().max().item(), 1)

        return wav.squeeze(0).squeeze(0).cpu().numpy()


    def get_estimate(self, noisy):
        torch.set_num_threads(1)
        with torch.no_grad():
            estimate = self.model(noisy)
            estimate = (1 - self.alpha) * estimate + self.alpha * noisy
        return estimate


    def save_wavs(self, estimates, noisy_sigs, filenames, out_dir, sr=16_000):
        # Write result
        # for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        #     filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        #     # write(noisy, filename + "_noisy.wav", sr=sr)
        #     self.write(estimate, filename+'.wav' , sr=sr)#+ "_enhanced.wav"

        file_name = os.path.basename(filenames).rsplit(".", 1)[0]
        self.write(estimates, os.path.join(out_dir,f'{file_name}_denoise.wav'), sr=sr)#+ "_enhanced.wav"


    def write(self, wav, filename, sr=16_000):
        # Normalize audio if it prevents clipping
        wav = wav / max(wav.abs().max().item(), 1)
        wav = wav.squeeze(0)
        torchaudio.save(filename, wav.cpu(), sr, encoding= "PCM_S", bits_per_sample=16)


    def get_dataset(self, in_dir, sample_rate, channels):
            
        files = find_audio_files(in_dir)

        return Audioset(files, with_path=True,
                        sample_rate=sample_rate, channels=channels, convert=True)


    def _estimate_and_save(self, noisy_signals, filenames, out_dir):
        estimate = self.get_estimate(noisy_signals)
        self.save_wavs(estimate, noisy_signals, filenames, out_dir, sr=self.model.sample_rate)


    def enhance_fullAudio(self, in_dir, out_dir):

        dset = self.get_dataset(in_dir, self.model.sample_rate, self.model.chin)
        if dset is None:
            return
        # loader = distrib.loader(dset, batch_size=1)
        
        with ProcessPoolExecutor(self.num_workers) as pool:
            print('Denoise starting...')
            # iterator = LogProgress(logger, loader, name="Generate enhanced files")
            pendings = []
            for data in dset:
                # Get batch data
                if data is not None:
                    noisy_signals, filenames = data
                    noisy_signals = noisy_signals.to(self.device)
                    # if self.device == 'cpu' and self.num_workers > 1:
                    #     pendings.append(
                    #         pool.submit(self._estimate_and_save,
                    #                     self.model, noisy_signals, filenames, out_dir))
                    # else:
                        # Forward
                    self._estimate_and_save(noisy_signals, filenames, out_dir)
                else:
                    return