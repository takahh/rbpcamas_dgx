# -------------------------------------------------------------------
# this code add padding and write with other data to make input files
# for the main program
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
import os
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
ipath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/past/protein/"
opath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/training_data_tokenized/all5_small/0.npz"
protein_cv_list_2805 = [['DDX6', 'SBDS', 'LIN28B', 'FTO', 'NOLC1', 'YBX3', 'SF3B1', 'CSTF2', 'IGF2BP1', 'DGCR8', 'SFPQ', 'NONO', 'SF3A3', 'SMNDC1', 'WDR43', 'LARP4', 'TAF15', 'PTBP1', 'NIPBL', 'XRCC6', 'SF3B4', 'CPEB4', 'NOL12', 'G3BP1', 'RPS3', 'U2AF1', 'AQR', 'GTF2F1', 'CPSF6', 'DDX3X'], ['PUS1', 'NPM1', 'DDX51', 'EIF4G2', 'HNRNPL', 'KHDRBS1', 'FAM120A', 'YWHAG', 'BCLAF1', 'CDC40', 'SRSF1', 'ZC3H8', 'DDX24', 'TARDBP', 'UTP18', 'AARS', 'EWSR1', 'BUD13', 'HNRNPK', 'U2AF2', 'FUBP3', 'SLTM', 'GRSF1', 'PABPN1', 'DROSHA', 'RBM22', 'STAU2', 'CSTF2T', 'DDX42', 'PRPF4'], ['SUB1', 'PCBP2', 'GPKOW', 'SLBP', 'AGGF1', 'PABPC4', 'SAFB2', 'TROVE2', 'SUGP2', 'TRA2A', 'GEMIN5', 'RPS11', 'PUM2', 'SRSF9', 'MATR3', 'DDX55', 'NKRF', 'AKAP8L', 'EIF3H', 'WRN', 'DKC1', 'PPIG', 'SAFB', 'TBRG4', 'LSM11', 'GRWD1', 'SRSF7', 'XRN2', 'AKAP1', 'SUPV3L1'], ['FUS', 'ZNF622', 'HNRNPA1', 'TIA1', 'QKI', 'UCHL5', 'GNL3', 'SDAD1', 'TIAL1', 'WDR3', 'MTPAP', 'DDX21', 'ZRANB2', 'LARP7', 'APOBEC3C', 'RBM15', 'FXR1', 'IGF2BP2', 'EXOSC5', 'HNRNPUL1', 'FKBP4', 'HNRNPU', 'NSUN2', 'HNRNPC', 'PPIL4', 'UTP3', 'ABCF1', 'EFTUD2', 'PRPF8', 'ZC3H11A'], ['EIF3D', 'IGF2BP3', 'SND1', 'SSB', 'PCBP1', 'METAP2', 'FMR1', 'NCBP2', 'RBFOX2', 'PUM1', 'HLTF', 'BCCIP', 'DHX30', 'FXR2', 'PHF6', 'RBM5', 'POLR2G', 'XPO5', 'FASTKD2', 'AATF', 'DDX52', 'UPF1', 'ZNF800', 'HNRNPM', 'EIF3G', 'NIP7', 'KHSRP', 'ILF3', 'SERBP1', 'DDX59']]

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def test():
    arr = np.load(f"{opath}", allow_pickle=True)
    print(arr.files)
    print(arr["proid"])
    iddict = get_pro_id_dict()
    print(iddict.keys())
    print(iddict)


def get_pro_id_dict():
    id_dict = {}
    lensum = 0
    for idx, item in enumerate(protein_cv_list_2805):  # idx=[0-4]
        for idx2, item2 in enumerate(item):
            id_dict[lensum + idx2] = item2
        lensum += len(item)
    return id_dict


def main():
    for files in os.listdir(ipath):
        if ".npy" in files:
            arr = np.load(f"{ipath}{files}", allow_pickle=True)
            print(arr.shape)


def checkif_zero_exists(n):
    for i in range(n):
        arr = np.load(opath.replace("0", str(i)), allow_pickle=True)
        # print(arr["proid"])
        if 0 in arr["proid"]:
            print(arr["proid"])
            return 1
    return 0


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # main()
    # test()
    # print(checkif_zero_exists(50))
