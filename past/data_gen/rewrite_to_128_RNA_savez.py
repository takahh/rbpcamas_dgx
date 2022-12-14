# testing how fast the loading two files

# for example, if in the group 0, training is 1-4, test is 0
# data is put to RNA128_training/0/ and RNA128_test/0/
# max data is 1,500,000 data for each
#             40,000 files

import numpy as np
import os

from sklearn.decomposition import PCA
import numpy as np
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

path = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/"
# path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
rna_from_dir = f"{path}finetune_data/"  # 0.npy
rna_to_train_dir = f"{path}RNA128_training_200/"  # 0.npy
rna_to_test_dir = f"{path}RNA128_test_200/"  # 0.npy
original_PROTNAMES = ['EIF3D', 'HNRNPL', 'BCLAF1', 'CPEB4', 'NSUN2', 'SMNDC1', 'AKAP8L', 'BCCIP', 'EIF3G', 'DDX24', 'METAP2', 'EIF4G2', 'LSM11', 'UCHL5', 'UTP18', 'AKAP1', 'SND1', 'DDX42', 'IGF2BP3', 'PTBP1', 'RBFOX2', 'GPKOW', 'SRSF7', 'STAU2', 'TAF15', 'SAFB2', 'ZC3H8', 'SF3B4', 'HNRNPA1', 'XRN2', 'SUGP2', 'XPO5', 'LIN28B', 'NONO', 'SDAD1', 'ZNF622', 'SFPQ', 'SF3B1', 'UTP3', 'CPSF6', 'LARP7', 'NIPBL', 'ZRANB2', 'KHSRP', 'WRN', 'HNRNPC', 'SRSF1', 'EIF3H', 'ILF3', 'PPIG', 'CSTF2T', 'LARP4', 'SSB', 'AGGF1', 'U2AF2', 'SF3A3', 'YWHAG', 'FMR1', 'U2AF1', 'NCBP2', 'DKC1', 'TRA2A', 'SERBP1', 'ZC3H11A', 'FXR2', 'FXR1', 'SUPV3L1', 'FKBP4', 'RPS11', 'GRSF1', 'ZNF800', 'SUB1', 'PPIL4', 'FUBP3', 'WDR3', 'NOLC1', 'UPF1', 'HNRNPU', 'HNRNPUL1', 'DROSHA', 'DDX52', 'QKI', 'TROVE2', 'DDX51', 'ABCF1', 'TBRG4', 'DDX21', 'XRCC6', 'GTF2F1', 'DGCR8', 'CDC40', 'MATR3', 'PHF6', 'POLR2G', 'NIP7', 'WDR43', 'DHX30', 'FUS', 'MTPAP', 'NPM1', 'HNRNPK', 'SRSF9', 'PABPC4', 'PRPF4', 'DDX3X', 'RPS3', 'AATF', 'SBDS', 'DDX59', 'RBM15', 'DDX6', 'PUM2', 'EXOSC5', 'HLTF', 'PUM1', 'PUS1', 'GEMIN5', 'GRWD1', 'RBM5', 'CSTF2', 'FTO', 'SLTM', 'GNL3', 'EWSR1', 'AARS', 'NKRF', 'FAM120A', 'PCBP2', 'FASTKD2', 'EFTUD2', 'PRPF8', 'RBM22', 'IGF2BP2', 'BUD13', 'SAFB', 'TIAL1', 'NOL12', 'IGF2BP1', 'PCBP1', 'SLBP', 'AQR', 'DDX55', 'APOBEC3C', 'G3BP1', 'KHDRBS1', 'PABPN1', 'TARDBP', 'HNRNPM', 'TIA1', 'YBX3', 'RPS5', 'TNRC6A']
PROTNAMES = ['U2AF2', 'EIF3D', 'XPO5', 'LIN28B', 'TNRC6A', 'SF3B4', 'PABPN1', 'FAM120A', 'STAU2', 'AKAP8L', 'PCBP2', 'DDX51', 'CPSF6', 'FXR1', 'RPS11', 'UTP3', 'MATR3', 'FASTKD2', 'TROVE2', 'UTP18', 'KHSRP', 'AQR', 'IGF2BP1', 'DDX52', 'BCCIP', 'DDX42', 'EIF4G2', 'SAFB2', 'RPS5', 'PRPF4', 'UCHL5', 'SSB', 'SLTM', 'SND1', 'PCBP1', 'WDR43', 'TIAL1', 'RBM22', 'ZC3H8', 'DHX30', 'HNRNPUL1', 'FMR1', 'ILF3', 'SRSF7', 'SRSF1', 'PUM2', 'GEMIN5', 'TAF15', 'DDX21', 'FUS', 'SFPQ', 'APOBEC3C', 'UPF1', 'HNRNPL', 'YBX3', 'PUM1', 'KHDRBS1', 'PRPF8', 'POLR2G', 'SUB1', 'GPKOW', 'LARP4', 'PPIL4', 'SRSF9', 'FTO', 'SLBP', 'ABCF1', 'QKI', 'YWHAG', 'EIF3H', 'FUBP3', 'SUPV3L1', 'SBDS', 'HNRNPA1', 'DDX59', 'GTF2F1', 'WRN', 'HNRNPC', 'AARS', 'EFTUD2', 'XRN2', 'PHF6', 'NCBP2', 'NSUN2', 'SF3A3', 'NKRF', 'SERBP1', 'GRSF1', 'NOL12', 'EWSR1', 'ZNF622', 'EXOSC5', 'TRA2A', 'NOLC1', 'TARDBP', 'NONO', 'HNRNPU', 'AATF', 'DDX24', 'HNRNPM', 'DDX6', 'FKBP4', 'AKAP1', 'ZC3H11A', 'CDC40', 'NIP7', 'CPEB4', 'AGGF1', 'METAP2', 'BUD13', 'WDR3', 'TIA1', 'CSTF2T', 'SMNDC1', 'DDX3X', 'DGCR8', 'NIPBL', 'U2AF1', 'CSTF2', 'SUGP2', 'PABPC4', 'RPS3', 'PUS1', 'SAFB', 'IGF2BP3', 'ZRANB2', 'SDAD1', 'HLTF', 'EIF3G', 'ZNF800', 'RBFOX2', 'IGF2BP2', 'GRWD1', 'DDX55', 'G3BP1', 'RBM15', 'TBRG4', 'SF3B1', 'FXR2', 'LSM11', 'DKC1', 'BCLAF1', 'PTBP1', 'GNL3', 'RBM5', 'DROSHA', 'MTPAP', 'HNRNPK', 'LARP7', 'NPM1', 'PPIG', 'XRCC6']


def getprotlist(group):
    eachlen = len(PROTNAMES)//4
    if group == 3:
        test_list = PROTNAMES[group * eachlen:]
    else:
        test_list = PROTNAMES[group * eachlen: (group + 1) * eachlen]
    train_list = [x for x in PROTNAMES if x not in test_list]
    return [train_list, test_list]


def load_array():
    rna_array = np.array([x[1] for x in np.load(f"{rna_from_dir}0.npy", allow_pickle=True)])
    rna_array = rna_array.reshape(5250, 768)
    return rna_array


def fit_pca(array_sample):
    pca = PCA(n_components=128)
    pca.fit(array_sample)
    return pca


def get_pca():
    rna_array = load_array()
    pca = fit_pca(rna_array)
    return pca


def store_and_write(first_or_not, proid_arr, label_arr, feature_arr, record, feat_vecs, file_num, group_num, path):
    if first_or_not == 0:
        proid_arr = np.concatenate([proid_arr, [record[0]]]).astype("float32")
        label_arr = np.concatenate([label_arr, [record[2]]]).astype("float32")
        feature_arr = np.concatenate([feature_arr, [feat_vecs]]).astype("float32")
        if proid_arr.shape[0] == 200:
            np.savez_compressed(f"{path}{group_num}/{file_num}", proid=proid_arr, feature=feature_arr, label=label_arr)
            proid_arr, feature_arr, label_arr = None, None, None
            first_or_not = 1
            file_num += 1
    elif first_or_not == 1:
        proid_arr = [record[0]]
        label_arr = [record[2]]
        feature_arr = [feat_vecs]
        first_or_not = 0
    return [first_or_not, proid_arr, label_arr, feature_arr, feat_vecs, file_num, group_num]


def get_rna_array():
    # make a pca function
    pca = get_pca()
    for j in range(0, 4):  # group
        x, y = 0, 0
        train_id_list, test_id_list = getprotlist(j)
        proid_arr_train, feature_arr_train, label_arr_train = None, None, None
        proid_arr_test, feature_arr_test, label_arr_test = None, None, None
        firsttrain, firsttest = 1, 1
        # read 1 file
        for i in range(10000):  # file
            for item in np.load(f"{rna_from_dir}{i}.npy", allow_pickle=True):  # data
                features = pca.transform(item[1])  # (105, 128)
                protein_name = original_PROTNAMES[abs(int(item[0]))]
                if protein_name in train_id_list:
                    firsttrain, proid_arr_train, label_arr_train, feature_arr_train, features, x, j =\
                        store_and_write(firsttrain, proid_arr_train, label_arr_train, feature_arr_train, item, features, x, j, rna_to_train_dir)
                elif protein_name in test_id_list:
                    firsttest, proid_arr_test, label_arr_test, feature_arr_test, features, y, j =\
                        store_and_write(firsttest, proid_arr_test, label_arr_test, feature_arr_test, item, features, y, j, rna_to_test_dir)
                else:
                    print("CONTINUE")
                    continue


if __name__ == "__main__":
    get_rna_array()
