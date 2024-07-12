import glob
import os
import pickle
import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch

# 确认有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx

def slice_motion(motion_file, stride, length, num_slices, out_dir):
    print(f"Motion file: {motion_file}")
    motion_data = pickle.load(open(motion_file, "rb"))
    pos, q = motion_data["pos"], motion_data["q"]
    scale = motion_data["scale"]

    print(f"Scale: {scale}, type: {type(scale)}")

    if isinstance(scale, list):
        scale = scale[0]

    pos /= scale
    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    start_idx = 0
    window = int(length * 30)  # assuming 30 FPS for motion data
    stride_step = int(stride * 30)
    slice_count = 0
    while start_idx <= len(pos) - window and slice_count < num_slices:
        pos_slice, q_slice = (
            pos[start_idx : start_idx + window],
            q[start_idx : start_idx + window],
        )
        out = {"pos": pos_slice, "q": q_slice}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count

def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    mismatched_files = []
    for wav, motion in tqdm(zip(wavs, motions), total=len(wavs)):
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
        if audio_slices != motion_slices:
            mismatched_files.append((wav, motion, audio_slices, motion_slices))
            print(f"Warning: Mismatch in slices for {m_name}: audio slices = {audio_slices}, motion slices = {motion_slices}")

    # 记录不匹配的文件
    with open("mismatched_files.txt", "w") as f:
        for wav, motion, audio_slices, motion_slices in mismatched_files:
            f.write(f"{wav}, {motion}, audio slices: {audio_slices}, motion slices: {motion_slices}\n")

def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        slice_audio(wav, stride, length, wav_out)
