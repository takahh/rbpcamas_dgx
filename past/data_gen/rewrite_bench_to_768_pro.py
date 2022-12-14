# testing how fast the loading two files

# returns
# 1. padded array
# 2. unpadded array
# 3. cross mask       3000 x 105  use unpadded array
# 4. cross mask small  105 x 105


import numpy as np
import os
from sklearn.decomposition import PCA
import numpy as np

# path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
path = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/"
pro_from_dir = f"{path}protein_bench/RPI369/"  # 1a1t-A.npy
pro_to_dir = f"{path}benchmark768_protein/RPI369/"  # 1a1t-A.npy
PROTEIN_SEQ_DIR = f"{path}benchmarks/sequence/RPI369_protein_seq.fa"  # AARS.npy


def make_protein_seq_dic_padded(path):
    pro_dict = {}
    for files in os.listdir(path):
        if "npy" in files:
            protname = files.split(".")[0]
            arrz = np.load(f"{path}{files}", allow_pickle=True)
            pro_dict[protname] = arrz["padded_array"]
    return pro_dict


def make_protein_seq_dic_unpadded(path):
    pro_dict = {}
    for files in os.listdir(path):
        if "npy" in files:
            protname = files.split(".")[0]
            arrz = np.load(f"{path}{files}", allow_pickle=True)
            pro_dict[protname] = arrz["unpadded_array"]
    return pro_dict


def get_pro_array():
    zero128 = [[0] * 768]
    for file in os.listdir(pro_from_dir):  # for each protein...
        if ".npy" not in file:
            continue
        # padding and pca application to raw data
        item = np.load(f"{pro_from_dir}{file}", allow_pickle=True)  # (1, 849, 768)
        new_array_formask = np.array([list(x) for x in item[0]])  # (849, 768) unpadded vecs
        if new_array_formask.shape[0] > 3000:
            print(new_array_formask.shape)
        # features_formask = pca.transform(new_array_formask)  # now the vectors are 128 dim
        features_formask = new_array_formask  # now the vectors are 128 dim
        new_array_toadd = np.array(features_formask, dtype=np.float32)  # (849, 128)
        padded_array_toadd = np.concatenate([new_array_toadd, zero128 * (4000 - new_array_toadd.shape[0])], axis=0)
        pro_padding_mask = np.concatenate([[0] * new_array_toadd.shape[1], [1] * (4000 - new_array_toadd.shape[1])])

        # # padded_array_toadd.shape (3000, 128)
        # cross_mask_array = np.concatenate([[pro_padding_mask] * 102, [[0] * 3000] * 3]).transpose().astype("float32")
        # cross_mask_array_small = np.concatenate([[[0] * 102 + [1] * 3] * 102, [[1] * 105] * 3]).astype("float32")

        new_array = new_array_toadd
        padded_array = np.array([padded_array_toadd], dtype="float32")

        np.savez_compressed(f"{pro_to_dir}{file}", padded_array=padded_array, unpadded_array=new_array)


def main():
    get_pro_array()


if __name__ == "__main__":
    main()
