import os
import argparse
import torch
import librosa
from glob import glob
from tqdm import tqdm

import utils
from wavlm import WavLM, WavLMConfig


def process(file_path):
    basename = os.path.basename(file_path)
    dirname = os.path.dirname(file_path)
    speaker = os.path.basename(dirname)
    
    save_dir = os.path.join(args.out_dir, speaker)
    os.makedirs(save_dir, exist_ok=True)
    
    save_name = os.path.join(save_dir, basename.replace(".wav", ".pt"))
    if not os.path.exists(save_name):
        wav, _ = librosa.load(file_path, sr=args.sr)
        wav = torch.from_numpy(wav).unsqueeze(0).cuda()
        c = utils.get_content(cmodel, wav)
        torch.save(c.cpu(), save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="dataset/universal-16k", help="path to input dir")
    parser.add_argument("--out_dir", type=str, default="dataset/wavlm", help="path to output dir")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading WavLM for content...")
    checkpoint = torch.load('wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")
    
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)
    
    for filename in tqdm(filenames):
        process(filename)
    