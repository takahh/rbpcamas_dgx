# -------------------------------------------------------------------
# this code tokenize sequences
# input : sequence
# output: array of tokens
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
import os
from zipfile import BadZipFile
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool
import multiprocessing as multi
import pickle

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# $$$$$$$$$$$$$$$ SEE HEREEEEEEEE!!!!!!!!!
moder = "all"  # shared or all
reduce_level = 8  # 20, 13, 8, 4
WRITE = 1
PDBID = "6V5B_C_D"
# $$$$$$$$$$$$$$$

group = "all5"
ogroup = "all5_small_tape"
path = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/"
input = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/batched_nparray/red{reduce_level}/"
group = f"{moder}_{reduce_level}"

# pot path and output path
path = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/attn_arrays_hb/all_8/"
hbpot_path = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/attn_arrays_hb/all_8/"
pipot_path = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/attn_arrays_pi/all_8/"

if not os.path.exists(output):
    os.mkdir(output)

# protein_cv_list = [['PPIL4', 'XRN2', 'PRPF4', 'SF3B1', 'NPM1', 'NIP7', 'RBM22', 'SRSF1', 'RPS5', 'EFTUD2', 'STAU2', 'KHDRBS1', 'DKC1', 'FKBP4', 'PUS1', 'YBX3', 'YWHAG', 'GTF2F1', 'KHSRP', 'DROSHA', 'LARP7', 'RPS11', 'HNRNPUL1', 'TRA2A', 'HNRNPC', 'UCHL5', 'LSM11', 'AATF', 'NOLC1', 'HNRNPK', 'ILF3'], ['NIPBL', 'RBFOX2', 'LIN28B', 'FXR1', 'TIA1', 'ZRANB2', 'APOBEC3C', 'IGF2BP1', 'BUD13', 'DHX30', 'GPKOW', 'NOL12', 'PABPN1', 'PCBP1', 'ZC3H11A', 'HLTF', 'SDAD1', 'RBM15', 'RBM27', 'RPS3', 'SRSF9', 'METAP2', 'CSTF2', 'U2AF2', 'NSUN2', 'AARS', 'EIF3D', 'TNRC6A', 'EIF3H', 'RBM5', 'DDX24'], ['HNRNPU', 'WRN', 'PPIG', 'SLTM', 'SERBP1', 'DGCR8', 'SUB1', 'EIF3G', 'EWSR1', 'SMNDC1', 'SAFB', 'TBRG4', 'DDX59', 'POLR2G', 'IGF2BP3', 'MTPAP', 'UTP3', 'BCLAF1', 'DDX3X', 'PTBP1', 'HNRNPA1', 'CPSF6', 'SAFB2', 'TROVE2', 'SUPV3L1', 'DDX21', 'DDX6', 'SUGP2', 'DDX42', 'XPO5', 'CSTF2T'], ['FUS', 'UTP18', 'SBDS', 'TAF15', 'NCBP2', 'CPEB4', 'WDR3', 'U2AF1', 'SF3B4', 'SF3A3', 'FTO', 'GRSF1', 'SLBP', 'HNRNPL', 'NKRF', 'EXOSC5', 'DDX55', 'TARDBP', 'ZC3H8', 'GRWD1', 'PUM1', 'EIF4G2', 'FMR1', 'SSB', 'XRCC6', 'G3BP1', 'AKAP8L', 'BCCIP', 'IGF2BP2', 'LARP4', 'FXR2'], ['PUM2', 'NONO', 'FASTKD2', 'PRPF8', 'PHF6', 'PABPC4', 'FAM120A', 'SFPQ', 'CDC40', 'HNRNPM', 'FUBP3', 'TIAL1', 'SRSF7', 'ZNF622', 'QKI', 'DDX52', 'GNL3', 'AGGF1', 'AKAP1', 'SND1', 'ZNF800', 'PCBP2', 'UPF1', 'MATR3', 'AUH', 'AQR', 'DDX51', 'ABCF1', 'GEMIN5', 'WDR43']]
PCODES_dict = {
        20: ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"],
        13: ['Y', 'L', 'K', 'G', 'W', 'I', 'R', 'F', 'C', 'H', 'D', 'A', 'N'],
        8: ['Y', 'G', 'I', 'R', 'N', 'F', 'H', 'K'],
        4: ['K', 'R', 'F', 'I']
}

PCODES = PCODES_dict[reduce_level]
RCODES = ["A", "G", "C", "U"]
ipath = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/past/protein/red{reduce_level}/"

# protein_cv_list_2805 = [['DDX6', 'SBDS', 'LIN28B', 'FTO', 'NOLC1', 'YBX3', 'SF3B1', 'CSTF2', 'IGF2BP1', 'DGCR8', 'SFPQ', 'NONO', 'SF3A3', 'SMNDC1', 'WDR43', 'LARP4', 'TAF15', 'PTBP1', 'NIPBL', 'XRCC6', 'SF3B4', 'CPEB4', 'NOL12', 'G3BP1', 'RPS3', 'U2AF1', 'AQR', 'GTF2F1', 'CPSF6', 'DDX3X'], ['PUS1', 'NPM1', 'DDX51', 'EIF4G2', 'HNRNPL', 'KHDRBS1', 'FAM120A', 'YWHAG', 'BCLAF1', 'CDC40', 'SRSF1', 'ZC3H8', 'DDX24', 'TARDBP', 'UTP18', 'AARS', 'EWSR1', 'BUD13', 'HNRNPK', 'U2AF2', 'FUBP3', 'SLTM', 'GRSF1', 'PABPN1', 'DROSHA', 'RBM22', 'STAU2', 'CSTF2T', 'DDX42', 'PRPF4'], ['SUB1', 'PCBP2', 'GPKOW', 'SLBP', 'AGGF1', 'PABPC4', 'SAFB2', 'TROVE2', 'SUGP2', 'TRA2A', 'GEMIN5', 'RPS11', 'PUM2', 'SRSF9', 'MATR3', 'DDX55', 'NKRF', 'AKAP8L', 'EIF3H', 'WRN', 'DKC1', 'PPIG', 'SAFB', 'TBRG4', 'LSM11', 'GRWD1', 'SRSF7', 'XRN2', 'AKAP1', 'SUPV3L1'], ['FUS', 'ZNF622', 'HNRNPA1', 'TIA1', 'QKI', 'UCHL5', 'GNL3', 'SDAD1', 'TIAL1', 'WDR3', 'MTPAP', 'DDX21', 'ZRANB2', 'LARP7', 'APOBEC3C', 'RBM15', 'FXR1', 'IGF2BP2', 'EXOSC5', 'HNRNPUL1', 'FKBP4', 'HNRNPU', 'NSUN2', 'HNRNPC', 'PPIL4', 'UTP3', 'ABCF1', 'EFTUD2', 'PRPF8', 'ZC3H11A'], ['EIF3D', 'IGF2BP3', 'SND1', 'SSB', 'PCBP1', 'METAP2', 'FMR1', 'NCBP2', 'RBFOX2', 'PUM1', 'HLTF', 'BCCIP', 'DHX30', 'FXR2', 'PHF6', 'RBM5', 'POLR2G', 'XPO5', 'FASTKD2', 'AATF', 'DDX52', 'UPF1', 'ZNF800', 'HNRNPM', 'EIF3G', 'NIP7', 'KHSRP', 'ILF3', 'SERBP1', 'DDX59']]
protein_cv_list_2805 = [['EIF3D', 'HNRNPL', 'BCLAF1', 'CPEB4', 'NSUN2', 'SMNDC1', 'AKAP8L', 'BCCIP', 'EIF3G', 'DDX24', 'METAP2', 'EIF4G2', 'LSM11', 'UCHL5', 'UTP18', 'AKAP1', 'SND1', 'DDX42', 'IGF2BP3', 'PTBP1', 'RBFOX2', 'GPKOW', 'SRSF7', 'STAU2', 'TAF15', 'SAFB2', 'ZC3H8', 'SF3B4', 'HNRNPA1', 'XRN2', 'SUGP2'], ['XPO5', 'LIN28B', 'NONO', 'SDAD1', 'ZNF622', 'SFPQ', 'SF3B1', 'UTP3', 'CPSF6', 'LARP7', 'NIPBL', 'ZRANB2', 'KHSRP', 'WRN', 'HNRNPC', 'SRSF1', 'EIF3H', 'ILF3', 'PPIG', 'CSTF2T', 'LARP4', 'SSB', 'AGGF1', 'U2AF2', 'SF3A3', 'YWHAG', 'FMR1', 'U2AF1', 'NCBP2', 'DKC1', 'TRA2A'], ['SERBP1', 'ZC3H11A', 'FXR2', 'FXR1', 'SUPV3L1', 'FKBP4', 'RPS11', 'GRSF1', 'ZNF800', 'SUB1', 'PPIL4', 'FUBP3', 'WDR3', 'NOLC1', 'UPF1', 'HNRNPU', 'HNRNPUL1', 'DROSHA', 'DDX52', 'QKI', 'AUH', 'TROVE2', 'DDX51', 'ABCF1', 'TBRG4', 'DDX21', 'XRCC6', 'GTF2F1', 'DGCR8', 'CDC40', 'MATR3'], ['PHF6', 'POLR2G', 'NIP7', 'WDR43', 'DHX30', 'FUS', 'MTPAP', 'NPM1', 'HNRNPK', 'SRSF9', 'PABPC4', 'PRPF4', 'DDX3X', 'RPS3', 'AATF', 'SBDS', 'DDX59', 'RBM15', 'DDX6', 'PUM2', 'EXOSC5', 'HLTF', 'RBM27', 'PUM1', 'PUS1', 'GEMIN5', 'GRWD1', 'RBM5', 'CSTF2', 'FTO', 'SLTM'], ['GNL3', 'EWSR1', 'AARS', 'NKRF', 'FAM120A', 'PCBP2', 'FASTKD2', 'EFTUD2', 'PRPF8', 'RBM22', 'IGF2BP2', 'BUD13', 'SAFB', 'TIAL1', 'NOL12', 'IGF2BP1', 'PCBP1', 'SLBP', 'AQR', 'DDX55', 'APOBEC3C', 'G3BP1', 'KHDRBS1', 'PABPN1', 'TARDBP', 'HNRNPM', 'TIA1', 'YBX3', 'RPS5', 'TNRC6A']]
# unknown_cv_list = 	[['SERBP1', 'KHDRBS1', 'SAFB2', 'SFPQ', 'CSTF2', 'ZC3H8', 'NSUN2', 'GTF2F1', 'ABCF1', 'POLR2G', 'LARP4', 'CDC40', 'DDX59', 'LSM11', 'PHF6', 'DDX6', 'TIAL1', 'IGF2BP1', 'DDX55', 'DDX42', 'GEMIN5', 'AUH', 'METAP2', 'SAFB', 'DKC1', 'SUB1', 'STAU2', 'SSB', 'FXR2', 'PCBP1', 'UTP18'], ['HNRNPU', 'ILF3', 'MATR3', 'CSTF2T', 'U2AF2', 'RPS11', 'NIP7', 'EIF3D', 'AGGF1', 'XPO5', 'DGCR8', 'FTO', 'FASTKD2', 'BUD13', 'RBM15', 'SF3B1', 'PUM2', 'SUPV3L1', 'PUS1', 'FXR1', 'DDX3X', 'EIF4G2', 'SRSF9', 'EIF3H', 'DDX21', 'FUS', 'RPS3', 'PABPC4', 'DDX24', 'DDX52', 'SDAD1'], ['HNRNPA1', 'KHSRP', 'HNRNPUL1', 'HNRNPM', 'TROVE2', 'XRN2', 'PCBP2', 'SND1', 'RBM5', 'CPEB4', 'U2AF1', 'XRCC6', 'UCHL5', 'HLTF', 'AKAP8L', 'RBM22', 'DHX30', 'TARDBP', 'FKBP4', 'GRWD1', 'ZNF622', 'IGF2BP3', 'TNRC6A', 'ZRANB2', 'IGF2BP2', 'PPIG', 'NOL12', 'BCLAF1', 'PUM1', 'WDR43', 'ZNF800'], ['HNRNPC', 'WDR3', 'TIA1', 'NONO', 'HNRNPK', 'AATF', 'NKRF', 'SLTM', 'PPIL4', 'DROSHA', 'TBRG4', 'NPM1', 'BCCIP', 'NCBP2', 'DDX51', 'PRPF8', 'UPF1', 'LARP7', 'SMNDC1', 'YWHAG', 'SRSF7', 'LIN28B', 'CPSF6', 'YBX3', 'TAF15', 'TRA2A', 'AQR', 'AKAP1', 'MTPAP', 'NIPBL', 'PRPF4'], ['SLBP', 'SBDS', 'SUGP2', 'QKI', 'RBFOX2', 'FUBP3', 'EXOSC5', 'EWSR1', 'FAM120A', 'EIF3G', 'PTBP1', 'SRSF1', 'NOLC1', 'WRN', 'EFTUD2', 'RPS5', 'UTP3', 'GPKOW', 'AARS', 'RBM27', 'GRSF1', 'SF3B4', 'HNRNPL', 'GNL3', 'ZC3H11A', 'SF3A3', 'PABPN1', 'FMR1', 'G3BP1', 'APOBEC3C']]

pmax = 2805

if moder == "unknown_shared_reduced_broad_share":
    filennum_per_group = 300
else:
    filennum_per_group = 1724

targetrange = list(range(filennum_per_group))
eachlen = round(filennum_per_group / 10)
promaxdict = {20: 2805, 13: 1609, 8: 986, 4: 609}
pmax_reduced = promaxdict[reduce_level]

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_pro_id_dict():
    id_dict = {}
    lensum = 0
    protein_list = protein_cv_list_2805
    for idx, item in enumerate(protein_list):  # idx=[0-4]
        for idx2, item2 in enumerate(item):
            id_dict[item2] = lensum + idx2
        lensum += len(item)
    return id_dict


def add_padding(arr, mode=None):
    if mode == "tape":
        max_len = promaxdict[reduce_level]
    else:
        max_len = 2805
    if arr.shape[1] > max_len - 1:
        return arr[:, :max_len, :]
    else:
        zerovec = np.array([[[0] * 768] * (max_len - arr.shape[1])])
        try:
            arr = np.concatenate((arr, zerovec), axis=1)
        except ValueError:
            print(zerovec.shape)
    return arr


def add_zero_pad_to_redidx(arr):
    zerovec = np.array([[0] * (2805 - arr[0].shape[0])])
    padded = np.concatenate([arr, zerovec], axis=1)
    return padded


def get_tape_out_dict():  # make dictionary of protein TAPE features
    proid_dict = get_pro_id_dict()
    tape_dict = {}
    for files in os.listdir(ipath):
        if ".npy" in files:
            pname = files.replace(".npy", "")
            if pname not in proid_dict.keys():
                continue
            arr = np.load(f"{ipath}{files}", allow_pickle=True)
            arr = arr[:, 1:-1, :]
            arr = add_padding(arr, "tape")
            if tape_dict == {}:
                tape_dict[proid_dict[pname]] = arr
            elif proid_dict[pname] not in tape_dict.keys():
                tape_dict[proid_dict[pname]] = arr
    return tape_dict


def get_p_tape_arr(pid_arr):
    arr = np.array([tape_dict[item] for item in enumerate(pid_arr)])
    return arr


def ptokenize(vocab_list, seq, lenmax):
    unpadded = [vocab_list.index(x) + 1 if x in vocab_list else 0 for x in seq]  # nonred
    pads = [0] * (lenmax - len(seq))  # pad as if the seq is reduced
    padded = unpadded + pads
    return padded


def rtokenize(vocab_list, seq):
    tokens = [vocab_list.index(x) + 1 if x in vocab_list else len(vocab_list) + 1 for x in seq]
    return tokens


def make_pmask(protein_length):
    first = np.repeat([0], repeats=protein_length, axis=0)
    second = np.repeat([1], repeats=(pmax_reduced - protein_length), axis=0)
    self_pro_mask_row = np.concatenate([first, second])
    upper_mask = np.repeat([self_pro_mask_row], repeats=protein_length, axis=0)
    lower_mask = np.repeat([[1] * pmax_reduced], repeats=(pmax_reduced - protein_length), axis=0)
    self_pro_mask_list = np.concatenate([upper_mask, lower_mask])
    self_pro_mask_list = self_pro_mask_list[np.newaxis, :, :]
    return self_pro_mask_list


def make_cross_mask(protein_length):
    first = np.repeat([0], repeats=protein_length, axis=0)
    second = np.repeat([1], repeats=(pmax_reduced - protein_length), axis=0)
    self_cross_mask_row = np.concatenate([first, second])
    cross_mask = np.repeat([self_cross_mask_row], repeats=101, axis=0)
    cross_mask = cross_mask[np.newaxis, :, :]
    return cross_mask


def make_label_vec(label):
    if label == 1:
        new_label_list = np.array([[0, 1]], dtype="float32")
    else:
        new_label_list = np.array([[1, 0]], dtype="float32")
    return new_label_list


def simple_main():
    for btype in ["hb", "pi"]:
        if btype == "hb":
            arr = np.load(f"{input}training_data/attn_analysis_hb/0.npy", allow_pickle=True)
            ogroup = "attn_analysis_hb"
        else:
            arr = np.load(f"{input}training_data/attn_analysis_hb/0.npy", allow_pickle=True)
            ogroup = "attn_analysis_pi"

        hbpot_arr = np.load(f"{hbpot_path}/0.npz", allow_pickle=True)["pot"]
        pipot_arr = np.load(f"{pipot_path}/0.npz", allow_pickle=True)["pot"]

        pro_id_arr, pro_token_arr, rna_token_arr, label_arr, pro_mask_arr, cross_mask_arr = None, None, None, None, None, None
        for idx, (item, hpot, ppot) in enumerate(zip(arr, hbpot_arr, pipot_arr)):
            # print(idx)
            pro_array_tokenized, pro_len = ptokenize(PCODES, item[1])
            rna_array_tokenized = rtokenize(RCODES, item[2])
            if idx == 0:
                pro_id_arr = np.atleast_2d([int(item[0])])
                p_tape_arr = tape_dict[int(item[0])]
                pro_token_arr = np.array(pro_array_tokenized)[np.newaxis, :]
                rna_token_arr = np.array(rna_array_tokenized)[np.newaxis, :]
                label_arr = make_label_vec(item[3])
                pro_mask_arr = make_pmask(pro_len)
                cross_mask_arr = make_cross_mask(pro_len)
                hpots = hpot[np.newaxis, :]
                ppots = ppot[np.newaxis, :]
            else:
                pro_id_arr = np.concatenate([pro_id_arr, np.atleast_2d([int(item[0])])], axis=0)
                p_tape_arr = np.concatenate([p_tape_arr, tape_dict[int(item[0])]], axis=0)
                pro_token_arr = np.concatenate([pro_token_arr, np.array(pro_array_tokenized)[np.newaxis, :]], axis=0)
                rna_token_arr = np.concatenate([rna_token_arr, np.array(rna_array_tokenized)[np.newaxis, :]], axis=0)
                label_arr = np.concatenate([label_arr, make_label_vec(item[3])], axis=0)
                pro_mask_arr = np.concatenate([pro_mask_arr, make_pmask(pro_len)], axis=0)
                cross_mask_arr = np.concatenate([cross_mask_arr, make_cross_mask(pro_len)], axis=0)
                hpots = np.concatenate([hpots, hpot[np.newaxis, :]], axis=0)
                ppots = np.concatenate([ppots, ppot[np.newaxis, :]], axis=0)
        # ogroup: "all5_small_tape" when known protein
        if not os.path.exists(f"{output}{ogroup}/{PDBID}"):
            os.mkdir(f"{output}{ogroup}/{PDBID}")
        if moder == "attn_ana":
            np.savez_compressed(f"{output}{ogroup}/{PDBID}/0", proid=pro_id_arr, protok=pro_token_arr, rnatok=rna_token_arr, p_tape_arr=p_tape_arr,
                                label=label_arr, pro_mask=pro_mask_arr, cross_mask=cross_mask_arr, hb_pots=hpots, pi_pots=ppots)
        else:
            np.savez_compressed(f"{output}{ogroup}/0", proid=pro_id_arr, protok=pro_token_arr, rnatok=rna_token_arr, p_tape_arr=p_tape_arr,
                                label=label_arr, pro_mask=pro_mask_arr, cross_mask=cross_mask_arr, hb_pots=hpots, pi_pots=ppots)


def main(gnum):  # gnum (gpu_id, protein_group_id)
    if moder == "shared":
        group = f"{moder}_{reduce_level}"
    else:
        print(f"gnum {gnum}")
        try:
            if len(gnum) == 2:
                group = gnum[1]  # 0-4
                gnum = gnum[0]   # 0-9
        except TypeError:
            pass
        else:
            if group is not None:  # when unknown protein, save to the group folder
                if not os.path.exists(f"{output}"):
                    os.mkdir(f"{output}")
    if gnum == 9:
        thisrange = targetrange[eachlen * gnum:]
    else:
        thisrange = targetrange[eachlen * gnum: eachlen * (gnum + 1)]
    print(f"thisrange {thisrange}")
    for i in thisrange:
        try:
            arr = np.load(f"{input}/{i}.npy", allow_pickle=True)
            hbpot_arr = np.load(f"{hbpot_path}/{i}.npz", allow_pickle=True)["pot"]
            pipot_arr = np.load(f"{pipot_path}/{i}.npz", allow_pickle=True)["pot"]
        except OSError:
            print("OSError")
            continue

        pro_id_arr, pro_token_arr, rna_token_arr, label_arr, pro_mask_arr, cross_mask_arr, red_idx_arr \
                , p_tape_arr, hpots, ppots, red_idx_arr, reduced_ptoks_arr = \
            None, None, None, None, None, None, None, None, None, None, None, None
        try:
            for idx, (item, hpot, ppot) in enumerate(zip(arr, hbpot_arr, pipot_arr)):
                # item = [proid, proseq, rseq, label, reducedproseq, reduce_index]
                pro_array_tokenized = ptokenize(PCODES, item[1], pmax)
                rna_array_tokenized = rtokenize(RCODES, item[2])
                reduced_ptoks = ptokenize(PCODES, item[4], pmax_reduced)
                pro_len = len(item[4])
                if idx == 0:
                    pro_id_arr = np.atleast_2d([int(item[0])])
                    p_tape_arr = tape_dict[int(item[0])]
                    pro_token_arr = np.array(pro_array_tokenized)[np.newaxis, :]
                    rna_token_arr = np.array(rna_array_tokenized)[np.newaxis, :]
                    label_arr = make_label_vec(item[3])
                    pro_mask_arr = make_pmask(pro_len)
                    cross_mask_arr = make_cross_mask(pro_len)
                    hpots = hpot[np.newaxis, :]
                    ppots = ppot[np.newaxis, :]
                    if gnum == 0:
                        print(f"ptoks {reduced_ptoks}")
                        print(f"hpots {hpots}")
                    reduced_ptoks_arr = np.array(reduced_ptoks)[np.newaxis, :]
                    red_idx_arr = add_zero_pad_to_redidx(np.array([item[5]]))
                else:
                    pro_id_arr = np.concatenate([pro_id_arr, np.atleast_2d([int(item[0])])], axis=0)
                    p_tape_arr = np.concatenate([p_tape_arr, tape_dict[int(item[0])]], axis=0)
                    pro_token_arr = np.concatenate([pro_token_arr, np.array(pro_array_tokenized)[np.newaxis, :]], axis=0)
                    rna_token_arr = np.concatenate([rna_token_arr, np.array(rna_array_tokenized)[np.newaxis, :]], axis=0)
                    label_arr = np.concatenate([label_arr, make_label_vec(item[3])], axis=0)
                    pro_mask_arr = np.concatenate([pro_mask_arr, make_pmask(pro_len)], axis=0)
                    cross_mask_arr = np.concatenate([cross_mask_arr, make_cross_mask(pro_len)], axis=0)
                    hpots = np.concatenate([hpots, hpot[np.newaxis, :]], axis=0)
                    ppots = np.concatenate([ppots, ppot[np.newaxis, :]], axis=0)
                    reduced_ptoks_arr = np.concatenate([reduced_ptoks_arr, np.array(reduced_ptoks)[np.newaxis, :]], axis=0)
                    red_idx_arr = np.concatenate([red_idx_arr, add_zero_pad_to_redidx(np.array([item[5]]))])
        except TypeError:
            pass
        # # ogroup: "all5_small_tape" when known protein
        output = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/mydata_red8_{}_{}/"
        if WRITE == 1:
            np.savez_compressed(f"{output}/{i}", proid=pro_id_arr, protok=pro_token_arr, rnatok=rna_token_arr, p_tape_arr=p_tape_arr,
                label=label_arr, pro_mask=pro_mask_arr, cross_mask=cross_mask_arr, hb_pots=hpots, pi_pots=ppots, reduced_index=red_idx_arr,
                            reduced_ptoks=reduced_ptoks_arr)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

tape_dict = get_tape_out_dict()
print(f"tape_dict.keys() {len(tape_dict.keys())}")

if __name__ == '__main__':
    p = Pool(10)
    p.map(main, range(10))
    p.close()
