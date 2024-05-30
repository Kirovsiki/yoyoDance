import glob
import os
import pickle
import shutil
from pathlib import Path

def fileToList(f):
    if not os.path.exists(f):
        return []
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out

filter_list = set(fileToList("./splits/ignore_list.txt"))
train_list = set(fileToList("./splits/crossmodal_train.txt"))
test_list = set(fileToList("./splits/crossmodal_test.txt"))

def split_data(dataset_path):
    # Check if motion and wav files exist
    motion_files = [os.path.join(dataset_path, "motions", f) for f in os.listdir(os.path.join(dataset_path, "motions"))]
    wav_files = [os.path.join(dataset_path, "wavs", f) for f in os.listdir(os.path.join(dataset_path, "wavs"))]

    for motion in motion_files:
        print(f"Checking file: {motion}")
        assert os.path.isfile(motion), f"File not found: {motion}"

    for wav in wav_files:
        print(f"Checking file: {wav}")
        assert os.path.isfile(wav), f"File not found: {wav}"

    # train - test split
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        Path(f"{split_name}/motions").mkdir(parents=True, exist_ok=True)
        Path(f"{split_name}/wavs").mkdir(parents=True, exist_ok=True)
        for sequence in split_list:
            if sequence in filter_list:
                continue
            motion = f"{dataset_path}/motions/{sequence}.pkl"
            wav = f"{dataset_path}/wavs/{sequence}.wav"
            assert os.path.isfile(motion), f"File not found: {motion}"
            assert os.path.isfile(wav), f"File not found: {wav}"
            motion_data = pickle.load(open(motion, "rb"))

            # 打印 motion_data 的所有键，调试用
            print(f"Keys in motion_data: {motion_data.keys()}")

            # 修改为新数据集的键名
            trans = motion_data.get("root_positions")
            pose = motion_data.get("rotations")
            scale = motion_data.get("smpl_scaling", 1)  # 如果没有 smpl_scaling，设置为默认值 1

            # 处理新的键
            out_data = {"pos": trans, "q": pose, "scale": scale}
            pickle.dump(out_data, open(f"{split_name}/motions/{sequence}.pkl", "wb"))
            shutil.copyfile(wav, f"{split_name}/wavs/{sequence}.wav")

if __name__ == "__main__":
    dataset_folder = "C:/Users/Administrator/Desktop/PhantomDanceDatav1.1"
    split_data(dataset_folder)
