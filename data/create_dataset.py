# import argparse
# import os
# from pathlib import Path

# from audio_extraction.baseline_features import \
#     extract_folder as baseline_extract
# from audio_extraction.jukebox_features import extract_folder as jukebox_extract
# from filter_split_data import *
# from slice import *


# def create_dataset(opt):
#     # split the data according to the splits files
#     print("Creating train / test split")
#     split_data(opt.dataset_folder)
#     # slice motions/music into sliding windows to create training dataset
#     print("Slicing train data")
#     slice_aistpp(f"train/motions", f"train/wavs")
#     print("Slicing test data")
#     slice_aistpp(f"test/motions", f"test/wavs")
#     # process dataset to extract audio features
#     if opt.extract_baseline:
#         print("Extracting baseline features")
#         baseline_extract("train/wavs_sliced", "train/baseline_feats")
#         baseline_extract("test/wavs_sliced", "test/baseline_feats")
#     if opt.extract_jukebox:
#         print("Extracting jukebox features")
#         jukebox_extract("train/wavs_sliced", "train/jukebox_feats")
#         jukebox_extract("test/wavs_sliced", "test/jukebox_feats")


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--stride", type=float, default=0.5)
#     parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
#     parser.add_argument(
#         "--dataset_folder",
#         type=str,
#         default="edge_aistpp",
#         help="folder containing motions and music",
#     )
#     parser.add_argument("--extract-baseline", action="store_true")
#     parser.add_argument("--extract-jukebox", action="store_true")
#     opt = parser.parse_args()
#     return opt


# if __name__ == "__main__":
#     opt = parse_opt()
#     create_dataset(opt)


# import argparse
# import os
# import glob
# import pickle
# import torch
# from torch.utils.data import DataLoader, Dataset
# import librosa as lr
# import numpy as np
# import soundfile as sf
# from tqdm import tqdm
# from audio_extraction.baseline_features import extract_folder as baseline_extract
# from audio_extraction.jukebox_features import extract_folder as jukebox_extract
# from filter_split_data import split_data

# # 确认有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class MotionDataset(Dataset):
#     def __init__(self, motion_files):
#         self.motion_files = motion_files

#     def __len__(self):
#         return len(self.motion_files)

#     def __getitem__(self, idx):
#         motion_file = self.motion_files[idx]
#         motion_data = pickle.load(open(motion_file, 'rb'))
#         return motion_data, motion_file

# def slice_audio(audio_file, stride, length, out_dir):
#     audio, sr = lr.load(audio_file, sr=None)
#     file_name = os.path.splitext(os.path.basename(audio_file))[0]
#     start_idx = 0
#     idx = 0
#     window = int(length * sr)
#     stride_step = int(stride * sr)
#     while start_idx <= len(audio) - window:
#         audio_slice = audio[start_idx: start_idx + window]
#         sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
#         start_idx += stride_step
#         idx += 1
#     return idx

# def slice_motion(motion, stride, length, motion_out):
#     print(f"Keys in motion_data: {motion.keys()}")

#     if "root_positions" not in motion or "rotations" not in motion:
#         print(f"Missing expected keys in motion data: {motion.keys()}")
#         return []

#     trans = torch.tensor(motion["root_positions"]).to(device)
#     pose = torch.tensor(motion["rotations"]).to(device)
#     bone_names = motion.get("bone_names", [])

#     num_frames = len(trans)
#     slices = []

#     window = int(length * 30)  # assuming 30 FPS for motion data
#     stride_step = int(stride * 30)

#     for start in range(0, num_frames - window + 1, stride_step):
#         end = start + window
#         motion_slice = {
#             "pos": trans[start:end].cpu().numpy(),
#             "q": pose[start:end].cpu().numpy(),
#             "bone_names": bone_names
#         }
#         slices.append(motion_slice)
#         output_filename = f"{motion_out}/{start:06d}.pkl"
#         with open(output_filename, 'wb') as f:
#             pickle.dump(motion_slice, f)

#     return slices

# def slice_aistpp(motion_dir, audio_dir, stride=0.5, length=5, batch_size=16):
#     motion_files = sorted(glob.glob(f"{motion_dir}/*.pkl"))
#     audio_files = sorted(glob.glob(f"{audio_dir}/*.wav"))

#     motion_out = motion_dir + "_sliced"
#     audio_out = audio_dir + "_sliced"

#     os.makedirs(motion_out, exist_ok=True)
#     os.makedirs(audio_out, exist_ok=True)

#     assert len(motion_files) == len(audio_files), "Mismatch between number of motion and audio files"

#     motion_dataset = MotionDataset(motion_files)
#     motion_loader = DataLoader(motion_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#     for i, (motions, motion_files_batch) in enumerate(tqdm(motion_loader, total=len(motion_loader))):
#         for motion_data, motion_file in zip(motions, motion_files_batch):
#             m_name = os.path.splitext(os.path.basename(motion_file))[0]
#             audio_file = os.path.join(audio_dir, m_name + ".wav")

#             assert os.path.isfile(audio_file), f"Audio file not found: {audio_file}"

#             audio_slices = slice_audio(audio_file, stride, length, audio_out)
#             motion_slices = slice_motion(motion_data, stride, length, motion_out)

#             if len(motion_slices) != audio_slices:
#                 print(f"Mismatch in slices for {m_name}: audio slices = {audio_slices}, motion slices = {len(motion_slices)}")

# def create_dataset(opt):
#     # 分割数据集
#     print("Creating train / test split")
#     split_data(opt.dataset_folder)
#     # 切片处理
#     print("Slicing train data")
#     slice_aistpp(f"{opt.dataset_folder}/train/motions", f"{opt.dataset_folder}/train/wavs", stride=opt.stride, length=opt.length)
#     print("Slicing test data")
#     slice_aistpp(f"{opt.dataset_folder}/test/motions", f"{opt.dataset_folder}/test/wavs", stride=opt.stride, length=opt.length)
#     # 提取音频特征
#     if opt.extract_baseline:
#         print("Extracting baseline features")
#         baseline_extract(f"{opt.dataset_folder}/train/wavs_sliced", f"{opt.dataset_folder}/train/baseline_feats")
#         baseline_extract(f"{opt.dataset_folder}/test/wavs_sliced", f"{opt.dataset_folder}/test/baseline_feats")
#     if opt.extract_jukebox:
#         print("Extracting jukebox features")
#         jukebox_extract(f"{opt.dataset_folder}/train/wavs_sliced", f"{opt.dataset_folder}/train/jukebox_feats")
#         jukebox_extract(f"{opt.dataset_folder}/test/wavs_sliced", f"{opt.dataset_folder}/test/jukebox_feats")

# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--stride", type=float, default=0.5)
#     parser.add_argument("--length", type=float, default=5.0, help="Window length in seconds")
#     parser.add_argument(
#         "--dataset_folder",
#         type=str,
#         default="edge_aistpp",
#         help="Folder containing motions and music",
#     )
#     parser.add_argument("--extract-baseline", action="store_true")
#     parser.add_argument("--extract-jukebox", action="store_true")
#     opt = parser.parse_args()
#     return opt

# if __name__ == "__main__":
#     opt = parse_opt()
#     create_dataset(opt)


import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import \
    extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from filter_split_data import *
from slice import *


def create_dataset(opt):
    # split the data according to the splits files
    print("Creating train / test split")
    split_data(opt.dataset_folder)
    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_aistpp(f"train/motions", f"train/wavs")
    print("Slicing test data")
    slice_aistpp(f"test/motions", f"test/wavs")
    # process dataset to extract audio features
    if opt.extract_baseline:
        print("Extracting baseline features")
        baseline_extract("train/wavs_sliced", "train/baseline_feats")
        baseline_extract("test/wavs_sliced", "test/baseline_feats")
    if opt.extract_jukebox:
        print("Extracting jukebox features")
        jukebox_extract("train/wavs_sliced", "train/jukebox_feats")
        jukebox_extract("test/wavs_sliced", "test/jukebox_feats")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)