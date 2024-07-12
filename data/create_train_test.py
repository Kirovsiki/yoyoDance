import os
from sklearn.model_selection import train_test_split

def generate_split_lists(dataset_folder, train_list_file, test_list_file, test_size=0.2):
    motion_files = [f for f in os.listdir(os.path.join(dataset_folder, "motions")) if f.endswith('.pkl')]
    motion_files = [os.path.splitext(f)[0] for f in motion_files]  # 去掉文件扩展名

    train_files, test_files = train_test_split(motion_files, test_size=test_size, random_state=42)

    with open(train_list_file, 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)

    with open(test_list_file, 'w') as f:
        for item in test_files:
            f.write("%s\n" % item)

# 使用函数生成分割列表
dataset_folder = "C:/Users/Administrator/Desktop/PhantomDanceDatav1.1"
generate_split_lists(
    dataset_folder=dataset_folder,
    train_list_file=r"C:\Users\Administrator\Desktop\EDGE\data\splits\crossmodal_train2.txt",
    test_list_file=r"C:\Users\Administrator\Desktop\EDGE\data\splits\crossmodal_test2.txt"
)
