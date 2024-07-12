import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from filter_split_data import split_data
from slice import slice_aistpp

def create_dataset(opt):
    # split the data according to the splits files
    print("Creating train / test split")
    split_data(opt.dataset_folder)
    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_aistpp(os.path.join(opt.dataset_folder, "train", "motions"), os.path.join(opt.dataset_folder, "train", "wavs"))
    print("Slicing test data")
    slice_aistpp(os.path.join(opt.dataset_folder, "test", "motions"), os.path.join(opt.dataset_folder, "test", "wavs"))
    # process dataset to extract audio features
    if opt.extract_baseline:
        print("Extracting baseline features")
        baseline_extract(os.path.join(opt.dataset_folder, "train", "wavs_sliced"), os.path.join(opt.dataset_folder, "train", "baseline_feats"))
        baseline_extract(os.path.join(opt.dataset_folder, "test", "wavs_sliced"), os.path.join(opt.dataset_folder, "test", "baseline_feats"))
    if opt.extract_jukebox:
        print("Extracting jukebox features")
        jukebox_extract(os.path.join(opt.dataset_folder, "train", "wavs_sliced"), os.path.join(opt.dataset_folder, "train", "jukebox_feats"))
        jukebox_extract(os.path.join(opt.dataset_folder, "test", "wavs_sliced"), os.path.join(opt.dataset_folder, "test", "jukebox_feats"))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="C:\\Users\\Administrator\\Desktop\\yoyodance_DATA",
        help="folder containing motions and music",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
