# import glob
# import os
# import pickle

# import librosa as lr
# import numpy as np
# import soundfile as sf
# from tqdm import tqdm


# def slice_audio(audio_file, stride, length, out_dir):
#     # stride, length in seconds
#     audio, sr = lr.load(audio_file, sr=None)
#     file_name = os.path.splitext(os.path.basename(audio_file))[0]
#     start_idx = 0
#     idx = 0
#     window = int(length * sr)
#     stride_step = int(stride * sr)
#     while start_idx <= len(audio) - window:
#         audio_slice = audio[start_idx : start_idx + window]
#         sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
#         start_idx += stride_step
#         idx += 1
#     return idx


# def slice_motion(motion_file, stride, length, num_slices, out_dir):
#     motion = pickle.load(open(motion_file, "rb"))
#     pos, q = motion["pos"], motion["q"]
#     scale = motion["scale"][0]

#     file_name = os.path.splitext(os.path.basename(motion_file))[0]
#     # normalize root position
#     pos /= scale
#     start_idx = 0
#     window = int(length * 60)
#     stride_step = int(stride * 60)
#     slice_count = 0
#     # slice until done or until matching audio slices
#     while start_idx <= len(pos) - window and slice_count < num_slices:
#         pos_slice, q_slice = (
#             pos[start_idx : start_idx + window],
#             q[start_idx : start_idx + window],
#         )
#         out = {"pos": pos_slice, "q": q_slice}
#         pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
#         start_idx += stride_step
#         slice_count += 1
#     return slice_count


# def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
#     wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
#     motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
#     wav_out = wav_dir + "_sliced"
#     motion_out = motion_dir + "_sliced"
#     os.makedirs(wav_out, exist_ok=True)
#     os.makedirs(motion_out, exist_ok=True)
#     assert len(wavs) == len(motions)
#     for wav, motion in tqdm(zip(wavs, motions)):
#         # make sure name is matching
#         m_name = os.path.splitext(os.path.basename(motion))[0]
#         w_name = os.path.splitext(os.path.basename(wav))[0]
#         assert m_name == w_name, str((motion, wav))
#         audio_slices = slice_audio(wav, stride, length, wav_out)
#         motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
#         # make sure the slices line up
#         assert audio_slices == motion_slices, str(
#             (wav, motion, audio_slices, motion_slices)
#         )


# def slice_audio_folder(wav_dir, stride=0.5, length=5):
#     wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
#     wav_out = wav_dir + "_sliced"
#     os.makedirs(wav_out, exist_ok=True)
#     for wav in tqdm(wavs):
#         audio_slices = slice_audio(wav, stride, length, wav_out)



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

def slice_motion(motion, stride, length, motion_out):
    print(f"Keys in motion_data: {motion.keys()}")

    if "root_positions" not in motion or "rotations" not in motion:
        print(f"Missing expected keys in motion data: {motion.keys()}")
        return []

    trans = motion["root_positions"]
    pose = motion["rotations"]
    bone_names = motion.get("bone_names", [])

    num_frames = len(trans)
    slices = []

    window = int(length * 30)  # assuming 30 FPS for motion data
    stride_step = int(stride * 30)

    for start in range(0, num_frames - window + 1, stride_step):
        end = start + window
        motion_slice = {
            "pos": torch.tensor(trans[start:end]).to(device),
            "q": torch.tensor(pose[start:end]).to(device),
            "bone_names": bone_names
        }
        slices.append(motion_slice)
        output_filename = f"{motion_out}/{start:06d}.pkl"
        with open(output_filename, 'wb') as f:
            pickle.dump(motion_slice, f)

    return slices

def slice_aistpp(motion_dir, audio_dir, stride=0.5, length=5):
    motion_files = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    audio_files = sorted(glob.glob(f"{audio_dir}/*.wav"))

    motion_out = motion_dir + "_sliced"
    audio_out = audio_dir + "_sliced"

    os.makedirs(motion_out, exist_ok=True)
    os.makedirs(audio_out, exist_ok=True)

    assert len(motion_files) == len(audio_files), "Mismatch between number of motion and audio files"

    for motion_file, audio_file in tqdm(zip(motion_files, audio_files), total=len(motion_files)):
        m_name = os.path.splitext(os.path.basename(motion_file))[0]
        a_name = os.path.splitext(os.path.basename(audio_file))[0]

        assert m_name == a_name, f"Mismatch between motion file {m_name} and audio file {a_name}"

        motion_data = pickle.load(open(motion_file, 'rb'))

        audio_slices = slice_audio(audio_file, stride, length, audio_out)
        motion_slices = slice_motion(motion_data, stride, length, motion_out)

        if len(motion_slices) != audio_slices:
            print(f"Mismatch in slices for {m_name}: audio slices = {audio_slices}, motion slices = {len(motion_slices)}")

def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        slice_audio(wav, stride, length, wav_out)
