# -------------------------------------------------------------------
# this code sees how much RNA sequences of a protein are shared with other proteins
# input : 
# output: 
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# this code writes token list of [RNA and protein and label]
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os, sys
import numpy as np
from Bio import pairwise2
from multiprocessing.pool import ThreadPool as Pool
import copy
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
group = 0
PALARELL = 1

# ------------------------  CHECK AND CHANGE HERE IF NECESSARY !!!!!!
mode = "all"  # all or lnc or all_unknown_protein
# ---------------------------------------

# input = f"/Users/mac/Documents/transformer_tape_dnabert/train_dir/"
# output = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/plot/shared_rna_seqs/count_data/"

input = f"/Users/mac/Documents/transformer_tape_dnabert/data/clusteredRBPsuite/82/"
output = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/plot/shared_rna_seqs/count_data/"

protein_cv_list_2805 = [['EIF3D', 'HNRNPL', 'BCLAF1', 'CPEB4', 'NSUN2', 'SMNDC1', 'AKAP8L', 'BCCIP', 'EIF3G', 'DDX24', 'METAP2', 'EIF4G2', 'LSM11', 'UCHL5', 'UTP18', 'AKAP1', 'SND1', 'DDX42', 'IGF2BP3', 'PTBP1', 'RBFOX2', 'GPKOW', 'SRSF7', 'STAU2', 'TAF15', 'SAFB2', 'ZC3H8', 'SF3B4', 'HNRNPA1', 'XRN2', 'SUGP2'], ['XPO5', 'LIN28B', 'NONO', 'SDAD1', 'ZNF622', 'SFPQ', 'SF3B1', 'UTP3', 'CPSF6', 'LARP7', 'NIPBL', 'ZRANB2', 'KHSRP', 'WRN', 'HNRNPC', 'SRSF1', 'EIF3H', 'ILF3', 'PPIG', 'CSTF2T', 'LARP4', 'SSB', 'AGGF1', 'U2AF2', 'SF3A3', 'YWHAG', 'FMR1', 'U2AF1', 'NCBP2', 'DKC1', 'TRA2A'], ['SERBP1', 'ZC3H11A', 'FXR2', 'FXR1', 'SUPV3L1', 'FKBP4', 'RPS11', 'GRSF1', 'ZNF800', 'SUB1', 'PPIL4', 'FUBP3', 'WDR3', 'NOLC1', 'UPF1', 'HNRNPU', 'HNRNPUL1', 'DROSHA', 'DDX52', 'QKI', 'AUH', 'TROVE2', 'DDX51', 'ABCF1', 'TBRG4', 'DDX21', 'XRCC6', 'GTF2F1', 'DGCR8', 'CDC40', 'MATR3'], ['PHF6', 'POLR2G', 'NIP7', 'WDR43', 'DHX30', 'FUS', 'MTPAP', 'NPM1', 'HNRNPK', 'SRSF9', 'PABPC4', 'PRPF4', 'DDX3X', 'RPS3', 'AATF', 'SBDS', 'DDX59', 'RBM15', 'DDX6', 'PUM2', 'EXOSC5', 'HLTF', 'RBM27', 'PUM1', 'PUS1', 'GEMIN5', 'GRWD1', 'RBM5', 'CSTF2', 'FTO', 'SLTM'], ['GNL3', 'EWSR1', 'AARS', 'NKRF', 'FAM120A', 'PCBP2', 'FASTKD2', 'EFTUD2', 'PRPF8', 'RBM22', 'IGF2BP2', 'BUD13', 'SAFB', 'TIAL1', 'NOL12', 'IGF2BP1', 'PCBP1', 'SLBP', 'AQR', 'DDX55', 'APOBEC3C', 'G3BP1', 'KHDRBS1', 'PABPN1', 'TARDBP', 'HNRNPM', 'TIA1', 'YBX3', 'RPS5', 'TNRC6A']]
flat_pro_list = [x for y in protein_cv_list_2805 for x in y]
proseq_master = f"/Users/mac/Documents/transformer_tape_dnabert/data/proteinseqs.csv"
print(flat_pro_list.index("SAFB"))


def clustered_to_one_group():  # [["SEDR", "AACU..."], []...]
    all_seq_dict = {}
    for proname in flat_pro_list:
        for dir in os.listdir(input):
            if ".DS" in dir:
                continue
            if "negative" in dir:
                if proname == dir.split("negative")[0]:
                    arr = np.load(f"{input}{dir}")
            elif "positive" in dir:
                if proname == dir.split("positive")[0]:
                    arr = np.load(f"{input}{dir}")
        all_seq_dict[proname] = arr
    return all_seq_dict


alldict = clustered_to_one_group()
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def seq_simlarity(data_dict):
    targetdict = copy.deepcopy(data_dict)
    for proname in data_dict.keys():
        count_list = []
        for seq in data_dict[proname]:
            protein_count = 0
            for target_proname in targetdict.keys():
                found = False
                if target_proname != proname:
                    for tar_seq in targetdict[target_proname]:
                        # if pairwise2.align.globalxx(seq, tar_seq)[0].score > thold:
                        if seq == tar_seq:
                            found = True
                            break
                if found is True:
                    protein_count += 1
                    break
                else:
                    pass
                # if gnumber == 0:
                #     print(f"{len(count_list)}")
            count_list.append(protein_count)
        if sum(count_list) != 0:
            print(f"{proname}: {sum(count_list)}")
        np.save(f"{output}{proname}", np.array(count_list))


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
def main():
    seq_simlarity(alldict)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()
