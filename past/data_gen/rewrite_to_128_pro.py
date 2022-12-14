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

path = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/"
# path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
pro_from_dir = f"{path}protein/"  # AARS.npy
pro_to_dir = f"{path}protein128/"  # AARS.npy
PROTNAMES = ['EIF3D', 'HNRNPL', 'BCLAF1', 'CPEB4', 'NSUN2', 'SMNDC1', 'AKAP8L', 'BCCIP', 'EIF3G', 'DDX24', 'METAP2', 'EIF4G2', 'LSM11', 'UCHL5', 'UTP18', 'AKAP1', 'SND1', 'DDX42', 'IGF2BP3', 'PTBP1', 'RBFOX2', 'GPKOW', 'SRSF7', 'STAU2', 'TAF15', 'SAFB2', 'ZC3H8', 'SF3B4', 'HNRNPA1', 'XRN2', 'SUGP2', 'XPO5', 'LIN28B', 'NONO', 'SDAD1', 'ZNF622', 'SFPQ', 'SF3B1', 'UTP3', 'CPSF6', 'LARP7', 'NIPBL', 'ZRANB2', 'KHSRP', 'WRN', 'HNRNPC', 'SRSF1', 'EIF3H', 'ILF3', 'PPIG', 'CSTF2T', 'LARP4', 'SSB', 'AGGF1', 'U2AF2', 'SF3A3', 'YWHAG', 'FMR1', 'U2AF1', 'NCBP2', 'DKC1', 'TRA2A', 'SERBP1', 'ZC3H11A', 'FXR2', 'FXR1', 'SUPV3L1', 'FKBP4', 'RPS11', 'GRSF1', 'ZNF800', 'SUB1', 'PPIL4', 'FUBP3', 'WDR3', 'NOLC1', 'UPF1', 'HNRNPU', 'HNRNPUL1', 'DROSHA', 'DDX52', 'QKI', 'TROVE2', 'DDX51', 'ABCF1', 'TBRG4', 'DDX21', 'XRCC6', 'GTF2F1', 'DGCR8', 'CDC40', 'MATR3', 'PHF6', 'POLR2G', 'NIP7', 'WDR43', 'DHX30', 'FUS', 'MTPAP', 'NPM1', 'HNRNPK', 'SRSF9', 'PABPC4', 'PRPF4', 'DDX3X', 'RPS3', 'AATF', 'SBDS', 'DDX59', 'RBM15', 'DDX6', 'PUM2', 'EXOSC5', 'HLTF', 'PUM1', 'PUS1', 'GEMIN5', 'GRWD1', 'RBM5', 'CSTF2', 'FTO', 'SLTM', 'GNL3', 'EWSR1', 'AARS', 'NKRF', 'FAM120A', 'PCBP2', 'FASTKD2', 'EFTUD2', 'PRPF8', 'RBM22', 'IGF2BP2', 'BUD13', 'SAFB', 'TIAL1', 'NOL12', 'IGF2BP1', 'PCBP1', 'SLBP', 'AQR', 'DDX55', 'APOBEC3C', 'G3BP1', 'KHDRBS1', 'PABPN1', 'TARDBP', 'HNRNPM', 'TIA1', 'YBX3', 'RPS5', 'TNRC6A']
BASE_PATH = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/"
PROTEIN_SEQ_FILE = f"{BASE_PATH}data/protein128/"  # AARS.npy

def make_protein_seq_dic_padded(path):
    pro_dict = {}
    for files in os.listdir(path):
        if "npz" in files:
            protname = files.split(".")[0]
            arrz = np.load(f"{path}{files}", allow_pickle=True)
            pro_dict[protname] = arrz["padded_array"]
    return pro_dict


def make_protein_seq_dic_unpadded(path):
    pro_dict = {}
    for files in os.listdir(path):
        if "npz" in files:
            protname = files.split(".")[0]
            arrz = np.load(f"{path}{files}", allow_pickle=True)
            pro_dict[protname] = arrz["unpadded_array"]
    return pro_dict


def get_pro_array():
    # make a pca function
    pro_array = np.load(f"{pro_from_dir}PRPF8.npy", allow_pickle=True)
    pca = PCA(n_components=128)
    new_array = np.array([list(x) for x in pro_array[0]])
    pca.fit(new_array)

    # apply pca
    zero128 = [[0] * 128]
    for file in os.listdir(pro_from_dir):  # for each protein...
        if ".npy" not in file:
            continue
        # padding and pca application to raw data
        item = np.load(f"{pro_from_dir}{file}", allow_pickle=True)  # (1, 849, 768)
        new_array_formask = np.array([list(x) for x in item[0]])  # (849, 768) unpadded vecs
        features_formask = pca.transform(new_array_formask)  # now the vectors are 128 dim
        new_array_toadd = np.array(features_formask, dtype=np.float32)  # (849, 128)
        padded_array_toadd = np.concatenate([new_array_toadd, zero128 * (3000 - new_array_toadd.shape[0])], axis=0)
        pro_padding_mask = np.concatenate([[0] * new_array_toadd.shape[1], [1] * (3000 - new_array_toadd.shape[1])])

        # padded_array_toadd.shape (3000, 128)
        cross_mask_array = np.concatenate([[pro_padding_mask] * 102, [[0] * 3000] * 3]).transpose().astype("float32")
        cross_mask_array_small = np.concatenate([[[0] * 102 + [1] * 3] * 102, [[1] * 105] * 3]).astype("float32")

        new_array = new_array_toadd
        padded_array = np.array([padded_array_toadd], dtype="float32")

        np.savez_compressed(f"{pro_to_dir}{file}", padded_array=padded_array, unpadded_array=new_array,
                cross_mask=cross_mask_array, cross_mask_small=cross_mask_array_small)


def main():
    get_pro_array()


if __name__ == "__main__":
    main()
