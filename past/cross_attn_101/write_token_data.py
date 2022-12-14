# -------------------------------------------------------------------
# this code writes token list of [RNA and protein and label]
# グループ０の物を１０個ずつくらい取り出して、NPARRAYにしてシャッフル、
# ３０個ずつ書き込む
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os, sys
import numpy as np

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
group = 0
count_per_file = 5
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
input = f"{path}RBPsuite_data/"
output = f"{path}training_data/"
output_one = f"{path}training_data/all5/"
# protein_cv_list = [['PPIL4', 'XRN2', 'PRPF4', 'SF3B1', 'NPM1', 'NIP7', 'RBM22', 'SRSF1', 'RPS5', 'EFTUD2', 'STAU2', 'KHDRBS1', 'DKC1', 'FKBP4', 'PUS1', 'YBX3', 'YWHAG', 'GTF2F1', 'KHSRP', 'DROSHA', 'LARP7', 'RPS11', 'HNRNPUL1', 'TRA2A', 'HNRNPC', 'UCHL5', 'LSM11', 'AATF', 'NOLC1', 'HNRNPK', 'ILF3'], ['NIPBL', 'RBFOX2', 'LIN28B', 'FXR1', 'TIA1', 'ZRANB2', 'APOBEC3C', 'IGF2BP1', 'BUD13', 'DHX30', 'GPKOW', 'NOL12', 'PABPN1', 'PCBP1', 'ZC3H11A', 'HLTF', 'SDAD1', 'RBM15', 'RBM27', 'RPS3', 'SRSF9', 'METAP2', 'CSTF2', 'U2AF2', 'NSUN2', 'AARS', 'EIF3D', 'TNRC6A', 'EIF3H', 'RBM5', 'DDX24'], ['HNRNPU', 'WRN', 'PPIG', 'SLTM', 'SERBP1', 'DGCR8', 'SUB1', 'EIF3G', 'EWSR1', 'SMNDC1', 'SAFB', 'TBRG4', 'DDX59', 'POLR2G', 'IGF2BP3', 'MTPAP', 'UTP3', 'BCLAF1', 'DDX3X', 'PTBP1', 'HNRNPA1', 'CPSF6', 'SAFB2', 'TROVE2', 'SUPV3L1', 'DDX21', 'DDX6', 'SUGP2', 'DDX42', 'XPO5', 'CSTF2T'], ['FUS', 'UTP18', 'SBDS', 'TAF15', 'NCBP2', 'CPEB4', 'WDR3', 'U2AF1', 'SF3B4', 'SF3A3', 'FTO', 'GRSF1', 'SLBP', 'HNRNPL', 'NKRF', 'EXOSC5', 'DDX55', 'TARDBP', 'ZC3H8', 'GRWD1', 'PUM1', 'EIF4G2', 'FMR1', 'SSB', 'XRCC6', 'G3BP1', 'AKAP8L', 'BCCIP', 'IGF2BP2', 'LARP4', 'FXR2'], ['PUM2', 'NONO', 'FASTKD2', 'PRPF8', 'PHF6', 'PABPC4', 'FAM120A', 'SFPQ', 'CDC40', 'HNRNPM', 'FUBP3', 'TIAL1', 'SRSF7', 'ZNF622', 'QKI', 'DDX52', 'GNL3', 'AGGF1', 'AKAP1', 'SND1', 'ZNF800', 'PCBP2', 'UPF1', 'MATR3', 'AUH', 'AQR', 'DDX51', 'ABCF1', 'GEMIN5', 'WDR43']]
protein_cv_list_2805 = [['DDX6', 'SBDS', 'LIN28B', 'FTO', 'NOLC1', 'YBX3', 'SF3B1', 'CSTF2', 'IGF2BP1', 'DGCR8', 'SFPQ', 'NONO', 'SF3A3', 'SMNDC1', 'WDR43', 'LARP4', 'TAF15', 'PTBP1', 'NIPBL', 'XRCC6', 'SF3B4', 'CPEB4', 'NOL12', 'G3BP1', 'RPS3', 'U2AF1', 'AQR', 'GTF2F1', 'CPSF6', 'DDX3X'], ['PUS1', 'NPM1', 'DDX51', 'EIF4G2', 'HNRNPL', 'KHDRBS1', 'FAM120A', 'YWHAG', 'BCLAF1', 'CDC40', 'SRSF1', 'ZC3H8', 'DDX24', 'TARDBP', 'UTP18', 'AARS', 'EWSR1', 'BUD13', 'HNRNPK', 'U2AF2', 'FUBP3', 'SLTM', 'GRSF1', 'PABPN1', 'DROSHA', 'RBM22', 'STAU2', 'CSTF2T', 'DDX42', 'PRPF4'], ['SUB1', 'PCBP2', 'GPKOW', 'SLBP', 'AGGF1', 'PABPC4', 'SAFB2', 'TROVE2', 'SUGP2', 'TRA2A', 'GEMIN5', 'RPS11', 'PUM2', 'SRSF9', 'MATR3', 'DDX55', 'NKRF', 'AKAP8L', 'EIF3H', 'WRN', 'DKC1', 'PPIG', 'SAFB', 'TBRG4', 'LSM11', 'GRWD1', 'SRSF7', 'XRN2', 'AKAP1', 'SUPV3L1'], ['FUS', 'ZNF622', 'HNRNPA1', 'TIA1', 'QKI', 'UCHL5', 'GNL3', 'SDAD1', 'TIAL1', 'WDR3', 'MTPAP', 'DDX21', 'ZRANB2', 'LARP7', 'APOBEC3C', 'RBM15', 'FXR1', 'IGF2BP2', 'EXOSC5', 'HNRNPUL1', 'FKBP4', 'HNRNPU', 'NSUN2', 'HNRNPC', 'PPIL4', 'UTP3', 'ABCF1', 'EFTUD2', 'PRPF8', 'ZC3H11A'], ['EIF3D', 'IGF2BP3', 'SND1', 'SSB', 'PCBP1', 'METAP2', 'FMR1', 'NCBP2', 'RBFOX2', 'PUM1', 'HLTF', 'BCCIP', 'DHX30', 'FXR2', 'PHF6', 'RBM5', 'POLR2G', 'XPO5', 'FASTKD2', 'AATF', 'DDX52', 'UPF1', 'ZNF800', 'HNRNPM', 'EIF3G', 'NIP7', 'KHSRP', 'ILF3', 'SERBP1', 'DDX59']]
protein_cv_list_1510 = [['DDX6', 'SBDS', 'LIN28B', 'FTO', 'NOLC1', 'YBX3', 'SF3B1', 'CSTF2', 'IGF2BP1', 'DGCR8', 'SFPQ', 'NONO', 'SF3A3', 'SMNDC1', 'WDR43', 'LARP4', 'TAF15', 'PTBP1',          'XRCC6', 'SF3B4', 'CPEB4', 'NOL12', 'G3BP1', 'RPS3', 'U2AF1', 'AQR', 'GTF2F1', 'CPSF6', 'DDX3X', 'PUS1'], ['NPM1', 'DDX51', 'EIF4G2', 'HNRNPL', 'KHDRBS1', 'FAM120A', 'YWHAG', 'BCLAF1', 'CDC40', 'SRSF1', 'ZC3H8', 'DDX24', 'TARDBP', 'UTP18', 'AARS', 'EWSR1', 'BUD13', 'HNRNPK', 'U2AF2', 'FUBP3', 'SLTM', 'GRSF1', 'PABPN1', 'DROSHA', 'RBM22', 'STAU2', 'CSTF2T', 'DDX42', 'PRPF4', 'SUB1'], ['PCBP2', 'GPKOW', 'SLBP', 'AGGF1', 'PABPC4', 'SAFB2', 'TROVE2', 'SUGP2', 'TRA2A', 'GEMIN5', 'RPS11', 'PUM2', 'SRSF9', 'MATR3', 'DDX55', 'NKRF', 'AKAP8L', 'EIF3H', 'WRN', 'DKC1', 'PPIG', 'SAFB', 'TBRG4', 'LSM11', 'GRWD1', 'SRSF7', 'XRN2', 'AKAP1', 'SUPV3L1', 'FUS'], ['ZNF622', 'HNRNPA1', 'TIA1', 'QKI', 'UCHL5', 'GNL3', 'SDAD1', 'TIAL1', 'WDR3', 'MTPAP', 'DDX21', 'ZRANB2', 'LARP7', 'APOBEC3C', 'RBM15', 'FXR1', 'IGF2BP2', 'EXOSC5', 'HNRNPUL1', 'FKBP4', 'HNRNPU', 'NSUN2', 'HNRNPC', 'PPIL4', 'UTP3', 'ABCF1', 'EFTUD2', 'ZC3H11A', 'EIF3D', 'IGF2BP3'], ['SND1', 'SSB', 'PCBP1', 'METAP2', 'FMR1', 'NCBP2', 'RBFOX2', 'PUM1', 'HLTF', 'BCCIP', 'DHX30', 'FXR2', 'PHF6', 'RBM5', 'POLR2G', 'XPO5', 'FASTKD2', 'AATF', 'DDX52', 'UPF1', 'ZNF800', 'HNRNPM', 'EIF3G', 'NIP7', 'KHSRP', 'ILF3', 'SERBP1', 'DDX59']]

proseq_master = f"/Users/mac/Documents/transformer_tape_dnabert/data/proteinseqs.csv"

# >chr12:13080717-13080852(+)
# ACCGUAAGUCAAAGGUCAGACUGUCCAGUGGGUGAUCUCUCAAGUCGCCCGCUUGGCCUCUUCCAAGUGUACUUUACUUCCUUUCAUUCCUGCUCUAAAAC
# >chrY:2357424-2357563(-)
# AGGUGGCCGAGAUGCUGUCGGCCUCCAGGUACCCCACCAUCAGCAUGGUGAAGCCGCUGCUGCACAUGCUCCUGAACACCACGCUCAACAUCAAGGAGACC

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_pro_id_dict():
    id_dict = {}
    lensum = 0
    for idx, item in enumerate(protein_cv_list_2805):  # idx=[0-4]
        for idx2, item2 in enumerate(item):
            id_dict[item2] = lensum + idx2
        lensum += len(item)
    return id_dict


def get_proseq(proname):
    sequence = None
    with open(proseq_master) as f:
        for lines in f.readlines():
            if proname in lines:
                sequence = lines.split(",")[1].strip()
    return sequence


def to_five_groups(group):
    datalist = []
    rnaseq = ""
    pro_id_dict = get_pro_id_dict()
    procount, file_count = 0, 0
    if not os.path.isdir(f"{output}{group}"):
        os.makedirs(f"{output}{group}")
    for dir in os.listdir(input):
        protein_name = dir.split(".")[0]
        if protein_name in protein_cv_list_2805[group]:
            proseq = get_proseq(protein_name)
            if proseq is None:
                print(protein_name)
                continue
            with open(f"{input}{dir}") as f:
                count = 0
                for lines in f.readlines():
                    print(count)
                    if count == 50:  # 50*30 = 1500 data = 300 files for posi/nega
                        break
                    elif ">" in lines:
                        if "negative" in dir:
                            label = 0
                        else:
                            label = 1
                    else:
                        rnaseq = lines.strip()
                        datalist.append([pro_id_dict[protein_name], proseq, rnaseq, label])
                        count += 1
            procount += 1

    arr = np.array(datalist)
    np.random.shuffle(arr)
    for i in range(arr.shape[0] // count_per_file):
        np.save(f"{output}{group}/{i}", arr[i * count_per_file: (i + 1) * count_per_file])


def to_one_group():
    datalist = []
    rnaseq = ""
    pro_id_dict = get_pro_id_dict()
    procount, file_count = 0, 0
    if not os.path.isdir(f"{output_one}"):
        os.makedirs(f"{output_one}")
    for dir in os.listdir(input):
        protein_name = dir.split(".")[0]
        proseq = get_proseq(protein_name)
        if proseq is None:
            print(protein_name)
            continue
        with open(f"{input}{dir}") as f:
            count = 0
            for lines in f.readlines():
                if count == 1000:
                    break
                elif ">" in lines:
                    if "negative" in dir:
                        label = 0
                    else:
                        label = 1
                else:
                    rnaseq = lines.strip()
                    datalist.append([pro_id_dict[protein_name], proseq, rnaseq, label])
                    count += 1
        procount += 1

    arr = np.array(datalist)
    np.random.shuffle(arr)
    for i in range(arr.shape[0] // count_per_file):
        if i != 45:
            continue
        np.save(f"{output_one}{i}", arr[i * count_per_file: (i + 1) * count_per_file])


def checkfiles():
    filename = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/training_data/0/0.npy"
    arr = np.load(filename)
    print(arr)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # for groupnum in range(5):
    #     to_five_groups(groupnum)
    # to_one_group()
    checkfiles()
