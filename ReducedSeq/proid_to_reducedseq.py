# -------------------------------------------------------------------
# this code writes token list of [RNA and protein and label]
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os, sys
import numpy as np

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------


one2three_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

reduce_resi_dict = {
    20: ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
          'SER', 'THR', 'TRP', 'TYR', 'VAL'],
    13: ['TYR', 'LEU', 'LYS', 'GLY', 'TRP', 'ILE', 'ARG', 'PHE', 'CYS', 'HIS', 'ASP', 'ALA', 'ASN'],
    8: ['TYR', 'GLY', 'ILE', 'ARG', 'ASN', 'PHE', 'HIS', 'LYS'],
    4: ['LYS', 'ARG', 'PHE', 'ILE']
}

group = 0

three2one_dict = {}
for item in one2three_dict.keys():
    three2one_dict[one2three_dict[item]] = item

np.random.seed(0)

# ------------------------  CHECK AND CHANGE HERE IF NECESSARY !!!!!!
mode = "unknown_protein_broad_share"  # all/lnc/all_unknown_protein/shared_RNA
REDUCE_LEVEL = 8  # 20, 13, 8, 4
# ---------------------------------------

count_per_file = 5
# input = f"/Users/mac/Documents/transformer_tape_dnabert/train_dir/"
# output = f"/Users/mac/Documents/transformer_tape_dnabert/data/unknown_protein/"

if mode == "shared_RNA": # shared across proteins
    input2 = f"/Users/mac/Documents/transformer_tape_dnabert/data/known_protein_rna_shared/58/"
    output_one = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/known_protein_reduced/shared_{REDUCE_LEVEL}"
elif mode == "unknown_protein":
    input2 = f"/Users/mac/Documents/transformer_tape_dnabert/data/known_protein_rna_shared/58/"
    output = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/unknown_reduced/"
    FILEPERGROUP = 660
elif mode == "unknown_protein_broad_share":
    input2 = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/selected_npy_per_RNA/inputdata/"
    output = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/unknown_reduced_broad_share/"
    FILEPERGROUP = 600

# if not os.path.exists(output_one):
#     os.mkdir(output_one)

# protein_cv_list_2805 = [['EIF3D', 'HNRNPL', 'BCLAF1', 'CPEB4', 'NSUN2', 'SMNDC1', 'AKAP8L', 'BCCIP', 'EIF3G', 'DDX24', 'METAP2', 'EIF4G2', 'LSM11', 'UCHL5', 'UTP18', 'AKAP1', 'SND1', 'DDX42', 'IGF2BP3', 'PTBP1', 'RBFOX2', 'GPKOW', 'SRSF7', 'STAU2', 'TAF15', 'SAFB2', 'ZC3H8', 'SF3B4', 'HNRNPA1', 'XRN2', 'SUGP2'], ['XPO5', 'LIN28B', 'NONO', 'SDAD1', 'ZNF622', 'SFPQ', 'SF3B1', 'UTP3', 'CPSF6', 'LARP7', 'NIPBL', 'ZRANB2', 'KHSRP', 'WRN', 'HNRNPC', 'SRSF1', 'EIF3H', 'ILF3', 'PPIG', 'CSTF2T', 'LARP4', 'SSB', 'AGGF1', 'U2AF2', 'SF3A3', 'YWHAG', 'FMR1', 'U2AF1', 'NCBP2', 'DKC1', 'TRA2A'], ['SERBP1', 'ZC3H11A', 'FXR2', 'FXR1', 'SUPV3L1', 'FKBP4', 'RPS11', 'GRSF1', 'ZNF800', 'SUB1', 'PPIL4', 'FUBP3', 'WDR3', 'NOLC1', 'UPF1', 'HNRNPU', 'HNRNPUL1', 'DROSHA', 'DDX52', 'QKI', 'AUH', 'TROVE2', 'DDX51', 'ABCF1', 'TBRG4', 'DDX21', 'XRCC6', 'GTF2F1', 'DGCR8', 'CDC40', 'MATR3'], ['PHF6', 'POLR2G', 'NIP7', 'WDR43', 'DHX30', 'FUS', 'MTPAP', 'NPM1', 'HNRNPK', 'SRSF9', 'PABPC4', 'PRPF4', 'DDX3X', 'RPS3', 'AATF', 'SBDS', 'DDX59', 'RBM15', 'DDX6', 'PUM2', 'EXOSC5', 'HLTF', 'RBM27', 'PUM1', 'PUS1', 'GEMIN5', 'GRWD1', 'RBM5', 'CSTF2', 'FTO', 'SLTM'], ['GNL3', 'EWSR1', 'AARS', 'NKRF', 'FAM120A', 'PCBP2', 'FASTKD2', 'EFTUD2', 'PRPF8', 'RBM22', 'IGF2BP2', 'BUD13', 'SAFB', 'TIAL1', 'NOL12', 'IGF2BP1', 'PCBP1', 'SLBP', 'AQR', 'DDX55', 'APOBEC3C', 'G3BP1', 'KHDRBS1', 'PABPN1', 'TARDBP', 'HNRNPM', 'TIA1', 'YBX3', 'RPS5', 'TNRC6A']]
unknown_cv_list = [['SERBP1', 'KHDRBS1', 'SAFB2', 'SFPQ', 'CSTF2', 'ZC3H8', 'NSUN2', 'GTF2F1', 'ABCF1', 'POLR2G', 'LARP4', 'CDC40', 'DDX59', 'LSM11', 'PHF6', 'DDX6', 'TIAL1', 'IGF2BP1', 'DDX55', 'DDX42', 'GEMIN5', 'AUH', 'METAP2', 'SAFB', 'DKC1', 'SUB1', 'STAU2', 'SSB', 'FXR2', 'PCBP1', 'UTP18'], ['HNRNPU', 'ILF3', 'MATR3', 'CSTF2T', 'U2AF2', 'RPS11', 'NIP7', 'EIF3D', 'AGGF1', 'XPO5', 'DGCR8', 'FTO', 'FASTKD2', 'BUD13', 'RBM15', 'SF3B1', 'PUM2', 'SUPV3L1', 'PUS1', 'FXR1', 'DDX3X', 'EIF4G2', 'SRSF9', 'EIF3H', 'DDX21', 'FUS', 'RPS3', 'PABPC4', 'DDX24', 'DDX52', 'SDAD1'], ['HNRNPA1', 'KHSRP', 'HNRNPUL1', 'HNRNPM', 'TROVE2', 'XRN2', 'PCBP2', 'SND1', 'RBM5', 'CPEB4', 'U2AF1', 'XRCC6', 'UCHL5', 'HLTF', 'AKAP8L', 'RBM22', 'DHX30', 'TARDBP', 'FKBP4', 'GRWD1', 'ZNF622', 'IGF2BP3', 'TNRC6A', 'ZRANB2', 'IGF2BP2', 'PPIG', 'NOL12', 'BCLAF1', 'PUM1', 'WDR43', 'ZNF800'], ['HNRNPC', 'WDR3', 'TIA1', 'NONO', 'HNRNPK', 'AATF', 'NKRF', 'SLTM', 'PPIL4', 'DROSHA', 'TBRG4', 'NPM1', 'BCCIP', 'NCBP2', 'DDX51', 'PRPF8', 'UPF1', 'LARP7', 'SMNDC1', 'YWHAG', 'SRSF7', 'LIN28B', 'CPSF6', 'YBX3', 'TAF15', 'TRA2A', 'AQR', 'AKAP1', 'MTPAP', 'NIPBL', 'PRPF4'], ['SLBP', 'SBDS', 'SUGP2', 'QKI', 'RBFOX2', 'FUBP3', 'EXOSC5', 'EWSR1', 'FAM120A', 'EIF3G', 'PTBP1', 'SRSF1', 'NOLC1', 'WRN', 'EFTUD2', 'RPS5', 'UTP3', 'GPKOW', 'AARS', 'RBM27', 'GRSF1', 'SF3B4', 'HNRNPL', 'GNL3', 'ZC3H11A', 'SF3A3', 'PABPN1', 'FMR1', 'G3BP1', 'APOBEC3C']]
proseq_master = f"/Users/mac/Documents/transformer_tape_dnabert/data/proteinseqs.csv"

# >chr12:13080717-13080852(+)
# ACCGUAAGUCAAAGGUCAGACUGUCCAGUGGGUGAUCUCUCAAGUCGCCCGCUUGGCCUCUUCCAAGUGUACUUUACUUCCUUUCAUUCCUGCUCUAAAAC
# >chrY:2357424-2357563(-)
# AGGUGGCCGAGAUGCUGUCGGCCUCCAGGUACCCCACCAUCAGCAUGGUGAAGCCGCUGCUGCACAUGCUCCUGAACACCACGCUCAACAUCAAGGAGACC

# get all protein names from RBPsuite file names
# all_protein_list = []
# for files in os.listdir(input):
#     protein_name = files.split(".")[0]
#     if protein_name not in all_protein_list:
#         all_protein_list.append(protein_name)
#
# # split the names into five groups
# protein_cv_list_2805 = []
# for i in range(5):
#     # try:
#     protein_cv_list_2805.append(all_protein_list[31*i: 31*(i+1)])
#
# print((protein_cv_list_2805))
# for item in protein_cv_list_2805:
#     print(len(item))

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_unknown_pro_id_dict():
    id = 0
    id_dict = {}
    for item in unknown_cv_list:
        for name in item:
            id_dict[name] = id
            id += 1
    return id_dict


def get_pro_id_dict():
    id = 0
    id_dict = {}
    for item in protein_cv_list_2805:
        for name in item:
            id_dict[name] = id
            id += 1
    return id_dict


def get_proseq(proname):
    sequence = None
    with open(proseq_master) as f:
        for lines in f.readlines():
            if proname == lines.split(",")[0]:
                sequence = lines.split(",")[1].strip()
    return sequence


def reduce_proseq(protein_seq):
    # target_index = []
    # for amino in reduce_resi_dict[REDUCE_LEVEL]:
    #     target_index += [idx for idx, x in enumerate(protein_seq) if amino == one2three_dict[x]]
    target_index = [idx for idx, x in enumerate(protein_seq) if one2three_dict[x] in reduce_resi_dict[REDUCE_LEVEL]]
    reducedseq = [x for x in protein_seq if one2three_dict[x] in reduce_resi_dict[REDUCE_LEVEL]]
    reducedseq = "".join(reducedseq)
    # target_index = sorted(target_index)
    return reducedseq, target_index


def to_five_groups():
    rnaseq = ""
    pro_id_dict = get_pro_id_dict()
    for group in range(5):
        count = 0
        datalist = []
        print(protein_cv_list_2805[group])
        print(len(protein_cv_list_2805[group]))
        if not os.path.isdir(f"{output}{group}"):
            os.makedirs(f"{output}{group}")
        for file in os.listdir(input):
            protein_name = file.split(".")[0]
            if protein_name in protein_cv_list_2805[group]:
                count += 1
                print(count)
                proseq = get_proseq(protein_name)
                # if count % 10 == 0:
                if proseq is None:
                    print(f"{protein_name} not in list!")
                    continue
                with open(f"{input}{file}") as f:
                    for lines in f.readlines():
                        if ">" in lines:
                            if "negative" in file:
                                label = 0
                            else:
                                label = 1
                        else:
                            rnaseq = lines.strip()
                            datalist.append([pro_id_dict[protein_name], proseq, rnaseq, label])

        # arr = np.array(datalist)
        # np.random.shuffle(arr)
        # for i in range(900):
        #     np.save(f"{output}{group}/{i}", arr[i * count_per_file: (i + 1) * count_per_file])


def unknown_to_five():
    pro_id_dict = get_unknown_pro_id_dict()
    for group in range(5):
        if not os.path.exists(f"{output}{group}"):
            os.mkdir(f"{output}{group}")
        filecount = 0
        final_arr = None
        for dir in os.listdir(input2):
            datalist = []
            if ".DS" in dir:
                continue
            if "nega" in dir:
                label = 0
                if "negative" in dir:
                    proname = dir.split("negative")[0]
                else:
                    proname = dir.split("_nega")[0]
            else:
                label = 1
                if "positive" in dir:
                    proname = dir.split("positive")[0]
                else:
                    proname = dir.split("_posi")[0]
            proid = pro_id_dict[proname]

            # if the protein belongs to the group, go on
            if proname in unknown_cv_list[group]:
                with open(f"{input2}{dir}") as f:
                    arr = [x.strip() for x in f.readlines()]
                # /Users/mac/Documents/transformer_tape_dnabert/data/known_protein_rna_shared/58/DDX55_posi.npy
                proseq = get_proseq(proname)
                # -----------------------------------------
                # get indices of reduced residues
                # -----------------------------------------
                reducedproseq, reduce_index = reduce_proseq(proseq)
                for item in arr:
                    datalist.append([proid, proseq, item, label, reducedproseq, reduce_index])
                arr = np.array(datalist)
                np.random.shuffle(arr)
                # arr = arr[0:17]
                if filecount == 0:
                    final_arr = arr
                else:
                    final_arr = np.concatenate([final_arr, arr])
                filecount += 1

        np.random.shuffle(final_arr)

        print(f"final_arr is {final_arr.shape}")
        for i in range(FILEPERGROUP):
            np.save(f"{output}{group}/{i}", final_arr[i * count_per_file: (i + 1) * count_per_file])


def to_one_group():
    for i in range(5):
        for j in range(900):
            arr = np.load(f"{output}{i}/{j}.npy")
            if i == 0 and j == 0:
                final_arr = arr
            else:
                final_arr = np.concatenate([final_arr, arr])

    np.random.shuffle(final_arr)
    for i in range(4500):
        # print(final_arr[i * count_per_file: (i + 1) * count_per_file].shape)
        np.save(f"{output_one}/{i}", final_arr[i * count_per_file: (i + 1) * count_per_file])


def clustered_to_one_group():
    pro_id_dict = get_pro_id_dict()
    count = 0
    for dir in os.listdir(input2):  #  e.g. AFR4_negative
        datalist = []
        if ".DS" in dir:
            continue
        arr = np.load(f"{input2}{dir}")
        # -----------------------------------------
        # obtain protein name from the file name
        # -----------------------------------------
        # shared: AARS_nega.npy, no-shared: AARSnegativefa_rep_seq.fasta.npy
        if "nega" in dir:
            label = 0
            if "negative" in dir:
                proname = dir.split("negative")[0]
            else:
                proname = dir.split("_nega")[0]
        else:
            label = 1
            if "positive" in dir:
                proname = dir.split("positive")[0]
            else:
                proname = dir.split("_posi")[0]
        # -----------------------------------------
        # get protein id and sequence
        # -----------------------------------------
        proid = pro_id_dict[proname]
        proseq = get_proseq(proname)
        # -----------------------------------------
        # get indices of reduced residues
        # -----------------------------------------
        reducedproseq, reduce_index = reduce_proseq(proseq)

        # -----------------------------------------
        # distribute the protein information (indices etc.)
        # -----------------------------------------
        for item in arr:
            datalist.append([proid, proseq, item, label, reducedproseq, reduce_index])
        arr = np.array(datalist)
        if count == 0:
            final_arr = arr
        else:
            final_arr = np.concatenate([final_arr, arr])
        count += 1
    itermax = final_arr.shape[0] // 5
    np.random.shuffle(final_arr)
    for i in range(itermax):
        if final_arr[i * count_per_file: (i + 1) * count_per_file].shape[0] != 5:
            print(f"final arr {i} - {final_arr[i * count_per_file: (i + 1) * count_per_file].shape}")
        np.save(f"{output_one}/{i}", final_arr[i * count_per_file: (i + 1) * count_per_file])


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # clustered_to_one_group()
    unknown_to_five()

    # # find smallest length of protein
    # iddict = get_pro_id_dict()
    # proseqlist = [get_proseq(x) for x in iddict.keys()]
    # reducedlenlist = sorted([len(reduce_proseq(x)[0]) for x in proseqlist], reverse=True)
    # print(reducedlenlist)
    # print(max(reducedlenlist))
