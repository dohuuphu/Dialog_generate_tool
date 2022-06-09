from pydub import AudioSegment
import glob
import tqdm

list_wav = glob.glob('/mnt/c/Users/phudh/Documents/data/mc_trangnghi/*')
for i in tqdm.tqdm(list_wav):
    result_path = i.replace('.mp3','.wav')
    sound = AudioSegment.from_mp3(i)
    sound.export(result_path, format="wav")
