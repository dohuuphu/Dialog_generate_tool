import torchaudio
from datasets import load_dataset, load_metric
# from transformers import (
#     Wav2Vec2ForCTC,
#     Wav2Vec2Processor,
# )
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import re
import sys

model_name = "base"
device = "cuda:1"
chars_to_ignore_regex = '[\\,\\?\\.\\!\\-\\;\\:\\"\\“\\%\\‘\\”\\�\\)\\(\\*)]'

model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

print("load data")
ds = load_dataset("./vi", "mt", split="test", data_dir="./vi")

resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)

def map_to_array(batch):
    speech, _ = torchaudio.load("./clips/"+batch["path"])
    batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
    batch["sampling_rate"] = resampler.new_freq
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

ds = ds.map(map_to_array)

def map_to_pred(batch):
    features = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0], padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["predicted"] = processor.batch_decode(pred_ids)
    batch["target"] = batch["sentence"]
    return batch

result = ds.map(map_to_pred, batched=True, batch_size=1, remove_columns=list(ds.features.keys()))

wer = load_metric("wer")
print(wer.compute(predictions=result["predicted"], references=result["target"]))
