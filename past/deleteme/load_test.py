# testing how fast the loading two files

import numpy as np
import os
from sklearn.decomposition import PCA
import numpy as np

# path = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/"
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
pro_from_dir = f"{path}protein/"  # AARS.npy
rna_from_dir = f"{path}finetune_data/"  # 0.npy
pro_to_dir = f"{path}protein128/AARS.npy"
rna_to_dir = f"{path}RNA128/"  # 0.npy


# rna test
 def get_rna_array():
    # # make a pca function
    # rna_array = np.array([x[1] for x in np.load(f"{rna_from_dir}0.npy", allow_pickle=True)])
    # rna_array = rna_array.reshape(5250, 768)
    # pca = PCA(n_components=128)
    # pca.fit(rna_array)
    #
    # # apply pca
    # for i in range(2002):
    #     array_list = np.array([])
    #     for item in np.load(f"{rna_from_dir}{i}.npy", allow_pickle=True):
    #         features = pca.transform(item[1])
    #         new_array = np.array([item[0], features, item[2]])
    #         if array_list.shape[0] != 0:
    #             array_list = np.concatenate([array_list, new_array])
    #         else:
    #             array_list = new_array
    #     np.save(f"{rna_to_dir}{i}.npy", array_list)

    # load test1
    for i in range(1):
        array = np.load(f"{rna_to_dir}{i}.npy", allow_pickle=True)
        print(array.shape)
        print("#######")
    # load test2
    for i in range(1):
        array = np.load(f"{rna_from_dir}{i}.npy", allow_pickle=True)
        print(array.shape)
    print("####")
    # return rna_array
    array = np.load(f"{pro_to_dir}", allow_pickle=True)
    print(array.shape)


def main():
    # dict_prot = make_protein_seq_dic(PROTEIN_SEQ_FILE)
    get_rna_array()


if __name__ == "__main__":
    main()
