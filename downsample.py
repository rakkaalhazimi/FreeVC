import os
import argparse
import glob
import itertools
from multiprocessing import Pool, cpu_count
from pathlib import PureWindowsPath

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


allowed_extensions = [".wav", ".ogg", ".mp3"]

parser = argparse.ArgumentParser()
parser.add_argument("--sr1", type=int, default=16000, help="sampling rate")
parser.add_argument("--sr2", type=int, default=22050, help="sampling rate")
parser.add_argument("--in_dir", type=str, default="./dataset/meme-indonesia", help="path to source dir")
parser.add_argument("--out_dir1", type=str, default="./dataset/universal-16k", help="path to target dir")
parser.add_argument("--out_dir2", type=str, default="./dataset/universal-22k", help="path to target dir")
args = parser.parse_args()


def process(wav_path):
    # speaker 's5', 'p280', 'p315' are excluded,
    wav_path = PureWindowsPath(wav_path).as_posix()
    wav_name = os.path.basename(wav_path)
    _, extension = os.path.splitext(wav_name)
    
    dirname = os.path.dirname(wav_path)
    speaker = os.path.basename(dirname)
    
    os.makedirs(os.path.join(args.out_dir1, speaker), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)
    wav, sr = librosa.load(wav_path)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = 0.98 * wav / peak
    wav1 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr1)
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
    save_name = wav_name.replace(extension, ".wav")
    save_path1 = os.path.join(args.out_dir1, speaker, save_name)
    save_path2 = os.path.join(args.out_dir2, speaker, save_name)
    wavfile.write(
        save_path1,
        args.sr1,
        (wav1 * np.iinfo(np.int16).max).astype(np.int16)
    )
    wavfile.write(
        save_path2,
        args.sr2,
        (wav2 * np.iinfo(np.int16).max).astype(np.int16)
    )


if __name__ == "__main__":
    
    pool = Pool(processes=cpu_count()-2)
    
    for ext in allowed_extensions:
        pattern = os.path.join(args.in_dir, f"**/*{ext}")
            
        for _ in tqdm(pool.imap_unordered(process, glob.glob(pattern, recursive=True))):
            pass
