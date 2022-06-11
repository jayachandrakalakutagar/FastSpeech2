import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    for file_name in (os.listdir(os.path.join(in_dir,"txt","p294"))):
        if file_name[-4:]!=".txt":
            continue
        base_name=file_name[:-4]
        text_path=os.path.join(in_dir,"txt","p294","{}.txt".format(base_name))
        wav_path=os.path.join(in_dir,"wav48","p294","{}.wav".format(base_name))

        with open(text_path) as f:
                text = f.readline().strip("\n")

        os.makedirs(os.path.join(out_dir, "p294"), exist_ok=True)
        wav, _ = librosa.load(wav_path, sampling_rate)
        wav = wav / max(abs(wav)) * 32768.0

        wavfile.write(
            os.path.join(out_dir, "p294", "{}.wav".format(base_name)),
            sampling_rate,
            wav.astype(np.int16),
        )
        with open(
            os.path.join(out_dir, "p294", "{}.lab".format(base_name)),
            "w",
         ) as f1:
            f1.write(text)