# making mask data for RNA for benchmarks

import numpy as np
import os

from sklearn.decomposition import PCA
import numpy as np
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

path = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/"
# path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
rna_from_dir = f"{path}finetune_bench/RPI369/"  # 0.npy
rna_to_dir = f"{path}benchmark768_RNA/RPI369/"  # 0.npy


def get_rna_array():
    zero768 = [[0] * 768]
    for file in os.listdir(rna_from_dir):  # for each protein...
        if ".npy" not in file:
            continue
        # padding and pca application to raw data
        features_formask = np.load(f"{rna_from_dir}{file}", allow_pickle=True)  # (1, 849, 768)
        # features_formask = np.array([list(x) for x in item])  # (849, 768) unpadded vecs
        print(features_formask.shape)
        new_array_toadd = np.array(features_formask, dtype=np.float32)  # (849, 128)
        padded_array_toadd = np.concatenate([new_array_toadd, zero768 * (4001 - new_array_toadd.shape[0])], axis=0)
        # pro_padding_mask = np.concatenate([[0] * new_array_toadd.shape[1], [1] * (4001 - new_array_toadd.shape[1])])

        new_array = new_array_toadd
        padded_array = np.array([padded_array_toadd], dtype="float32")
        np.savez_compressed(f"{rna_to_dir}{file}", padded_array=padded_array, unpadded_array=new_array)


if __name__ == "__main__":
    get_rna_array()
