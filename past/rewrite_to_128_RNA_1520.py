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
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

path = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/"
# path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
pro_from_dir = f"{path}protein/"  # AARS.npy
rna_from_dir = f"{path}finetune_data/"  # 0.npy
pro_to_dir = f"{path}finetune_data128/"  # AARS.npy
# rna_to_dir = f"{path}RNA128/"  # 0.npy
rna_to_train_dir = f"{path}RNA128_training_1520/"  # 0.npy
rna_to_test_dir = f"{path}RNA128_test_1520/"  # 0.npy

# a list, removed longest two chains GEMINI5 and NIPBL
PROTNAMES = ['EIF3D', 'HNRNPL', 'BCLAF1', 'CPEB4', 'NSUN2', 'SMNDC1', 'AKAP8L', 'BCCIP', 'EIF3G', 'DDX24', 'METAP2', 'EIF4G2', 'LSM11', 'UCHL5', 'UTP18', 'AKAP1', 'SND1', 'DDX42', 'IGF2BP3', 'PTBP1', 'RBFOX2', 'GPKOW', 'SRSF7', 'STAU2', 'TAF15', 'SAFB2', 'ZC3H8', 'SF3B4', 'HNRNPA1', 'XRN2', 'SUGP2', 'XPO5', 'LIN28B', 'NONO', 'SDAD1', 'ZNF622', 'SFPQ', 'SF3B1', 'UTP3', 'CPSF6', 'LARP7', 'ZRANB2', 'KHSRP', 'WRN', 'HNRNPC', 'SRSF1', 'EIF3H', 'ILF3', 'PPIG', 'CSTF2T', 'LARP4', 'SSB', 'AGGF1', 'U2AF2', 'SF3A3', 'YWHAG', 'FMR1', 'U2AF1', 'NCBP2', 'DKC1', 'TRA2A', 'SERBP1', 'ZC3H11A', 'FXR2', 'FXR1', 'SUPV3L1', 'FKBP4', 'RPS11', 'GRSF1', 'ZNF800', 'SUB1', 'PPIL4', 'FUBP3', 'WDR3', 'NOLC1', 'UPF1', 'HNRNPU', 'HNRNPUL1', 'DROSHA', 'DDX52', 'QKI', 'TROVE2', 'DDX51', 'ABCF1', 'TBRG4', 'DDX21', 'XRCC6', 'GTF2F1', 'DGCR8', 'CDC40', 'MATR3', 'PHF6', 'POLR2G', 'NIP7', 'WDR43', 'DHX30', 'FUS', 'MTPAP', 'NPM1', 'HNRNPK', 'SRSF9', 'PABPC4', 'PRPF4', 'DDX3X', 'RPS3', 'AATF', 'SBDS', 'DDX59', 'RBM15', 'DDX6', 'PUM2', 'EXOSC5', 'HLTF', 'PUM1', 'PUS1', 'GRWD1', 'RBM5', 'CSTF2', 'FTO', 'SLTM', 'GNL3', 'EWSR1', 'AARS', 'NKRF', 'FAM120A', 'PCBP2', 'FASTKD2', 'EFTUD2', 'PRPF8', 'RBM22', 'IGF2BP2', 'BUD13', 'SAFB', 'TIAL1', 'NOL12', 'IGF2BP1', 'PCBP1', 'SLBP', 'AQR', 'DDX55', 'APOBEC3C', 'G3BP1', 'KHDRBS1', 'PABPN1', 'TARDBP', 'HNRNPM', 'TIA1', 'YBX3', 'RPS5', 'TNRC6A']


def getprotlist(group):
    eachlen = len(PROTNAMES)//5 + 1
    test_list = PROTNAMES[group * eachlen: (group + 1) * eachlen]
    train_list = [x for x in PROTNAMES if x not in test_list]
    return [train_list, test_list]


def get_rna_array():
    # make a pca function
    rna_array = np.array([x[1] for x in np.load(f"{rna_from_dir}0.npy", allow_pickle=True)])
    rna_array = rna_array.reshape(5250, 768)
    pca = PCA(n_components=128)
    pca.fit(rna_array)


    # for j in range(5):
    for j in range(0, 5):
        x, y = 0, 0
        train_id_list, test_id_list = getprotlist(j)
        # read 1 file
        for i in range(10081):
            train_list = None
            test_list = None
            for item in np.load(f"{rna_from_dir}{i}.npy", allow_pickle=True):
                features = pca.transform(item[1])
                new_array = np.array([[item[0], features, item[2]]])
                if PROTNAMES[int(item[0])] in train_id_list:
                    if train_list is not None:
                        train_list = np.concatenate([train_list, new_array])
                    else:
                        train_list = new_array
                else:
                    if test_list is not None:
                        test_list = np.concatenate([test_list, new_array])
                    else:
                        test_list = new_array
            #  write training set
            for k in range(len(train_list) // 10):
                np.save(f"{rna_to_train_dir}{j}/{x}.npy", train_list[10 * k: 10 * (k + 1)])
                x += 1
            #  write test set
            if test_list is not None:
                for k in range(len(test_list) // 5):
                    np.save(f"{rna_to_test_dir}{j}/{y}.npy", test_list[5 * k: 5 * (k + 1)])
                    y += 1
            # break

    # # load test1
    # for i in range(1):
    #     array = np.load(f"{rna_to_dir}{i}.npy", allow_pickle=True)
    #     print(array.shape)
    #     print("#######")
    # # load test2
    # for i in range(1):
    #     array = np.load(f"{rna_from_dir}{i}.npy", allow_pickle=True)
    #     print(array.shape)

    # return rna_array


if __name__ == "__main__":
    get_rna_array()
