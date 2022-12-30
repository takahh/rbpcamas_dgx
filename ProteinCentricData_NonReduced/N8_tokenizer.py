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
from subprocess import call
from multiprocessing import Pool
from ProteinCentricData_reduced.S1_find_hotspots import makedir_if_not_exist

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# $$$$$$$$$$$$$$$ SEE HEREEEEEEEE!!!!!!!!!
moder = "all"  # shared or all
WRITE = 1
reduce_level = 20
# $$$$$$$$$$$$$$$

# pot path and output path
input = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/batched_nparray/nored/"
hbpot_path = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/nonreduced/attn_arrays_hb/"
pipot_path = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/nonreduced/attn_arrays_pi/"

# protein_cv_list = [['PPIL4', 'XRN2', 'PRPF4', 'SF3B1', 'NPM1', 'NIP7', 'RBM22', 'SRSF1', 'RPS5', 'EFTUD2', 'STAU2', 'KHDRBS1', 'DKC1', 'FKBP4', 'PUS1', 'YBX3', 'YWHAG', 'GTF2F1', 'KHSRP', 'DROSHA', 'LARP7', 'RPS11', 'HNRNPUL1', 'TRA2A', 'HNRNPC', 'UCHL5', 'LSM11', 'AATF', 'NOLC1', 'HNRNPK', 'ILF3'], ['NIPBL', 'RBFOX2', 'LIN28B', 'FXR1', 'TIA1', 'ZRANB2', 'APOBEC3C', 'IGF2BP1', 'BUD13', 'DHX30', 'GPKOW', 'NOL12', 'PABPN1', 'PCBP1', 'ZC3H11A', 'HLTF', 'SDAD1', 'RBM15', 'RBM27', 'RPS3', 'SRSF9', 'METAP2', 'CSTF2', 'U2AF2', 'NSUN2', 'AARS', 'EIF3D', 'TNRC6A', 'EIF3H', 'RBM5', 'DDX24'], ['HNRNPU', 'WRN', 'PPIG', 'SLTM', 'SERBP1', 'DGCR8', 'SUB1', 'EIF3G', 'EWSR1', 'SMNDC1', 'SAFB', 'TBRG4', 'DDX59', 'POLR2G', 'IGF2BP3', 'MTPAP', 'UTP3', 'BCLAF1', 'DDX3X', 'PTBP1', 'HNRNPA1', 'CPSF6', 'SAFB2', 'TROVE2', 'SUPV3L1', 'DDX21', 'DDX6', 'SUGP2', 'DDX42', 'XPO5', 'CSTF2T'], ['FUS', 'UTP18', 'SBDS', 'TAF15', 'NCBP2', 'CPEB4', 'WDR3', 'U2AF1', 'SF3B4', 'SF3A3', 'FTO', 'GRSF1', 'SLBP', 'HNRNPL', 'NKRF', 'EXOSC5', 'DDX55', 'TARDBP', 'ZC3H8', 'GRWD1', 'PUM1', 'EIF4G2', 'FMR1', 'SSB', 'XRCC6', 'G3BP1', 'AKAP8L', 'BCCIP', 'IGF2BP2', 'LARP4', 'FXR2'], ['PUM2', 'NONO', 'FASTKD2', 'PRPF8', 'PHF6', 'PABPC4', 'FAM120A', 'SFPQ', 'CDC40', 'HNRNPM', 'FUBP3', 'TIAL1', 'SRSF7', 'ZNF622', 'QKI', 'DDX52', 'GNL3', 'AGGF1', 'AKAP1', 'SND1', 'ZNF800', 'PCBP2', 'UPF1', 'MATR3', 'AUH', 'AQR', 'DDX51', 'ABCF1', 'GEMIN5', 'WDR43']]
PCODES_dict = {
        20: ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"],
        13: ['Y', 'L', 'K', 'G', 'W', 'I', 'R', 'F', 'C', 'H', 'D', 'A', 'N'],
        8: ['Y', 'G', 'I', 'R', 'N', 'F', 'H', 'K'],
        4: ['K', 'R', 'F', 'I']
}

PCODES = PCODES_dict[reduce_level]
RCODES = ["A", "G", "C", "U"]
ipath = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/past/protein/"  # TAPE features
protein_cv_list_2805 = [['EIF3D', 'HNRNPL', 'BCLAF1', 'CPEB4', 'NSUN2', 'SMNDC1', 'AKAP8L', 'BCCIP', 'EIF3G', 'DDX24', 'METAP2', 'EIF4G2', 'LSM11', 'UCHL5', 'UTP18', 'AKAP1', 'SND1', 'DDX42', 'IGF2BP3', 'PTBP1', 'RBFOX2', 'GPKOW', 'SRSF7', 'STAU2', 'TAF15', 'SAFB2', 'ZC3H8', 'SF3B4', 'HNRNPA1', 'XRN2', 'SUGP2'], ['XPO5', 'LIN28B', 'NONO', 'SDAD1', 'ZNF622', 'SFPQ', 'SF3B1', 'UTP3', 'CPSF6', 'LARP7', 'NIPBL', 'ZRANB2', 'KHSRP', 'WRN', 'HNRNPC', 'SRSF1', 'EIF3H', 'ILF3', 'PPIG', 'CSTF2T', 'LARP4', 'SSB', 'AGGF1', 'U2AF2', 'SF3A3', 'YWHAG', 'FMR1', 'U2AF1', 'NCBP2', 'DKC1', 'TRA2A'], ['SERBP1', 'ZC3H11A', 'FXR2', 'FXR1', 'SUPV3L1', 'FKBP4', 'RPS11', 'GRSF1', 'ZNF800', 'SUB1', 'PPIL4', 'FUBP3', 'WDR3', 'NOLC1', 'UPF1', 'HNRNPU', 'HNRNPUL1', 'DROSHA', 'DDX52', 'QKI', 'AUH', 'TROVE2', 'DDX51', 'ABCF1', 'TBRG4', 'DDX21', 'XRCC6', 'GTF2F1', 'DGCR8', 'CDC40', 'MATR3'], ['PHF6', 'POLR2G', 'NIP7', 'WDR43', 'DHX30', 'FUS', 'MTPAP', 'NPM1', 'HNRNPK', 'SRSF9', 'PABPC4', 'PRPF4', 'DDX3X', 'RPS3', 'AATF', 'SBDS', 'DDX59', 'RBM15', 'DDX6', 'PUM2', 'EXOSC5', 'HLTF', 'RBM27', 'PUM1', 'PUS1', 'GEMIN5', 'GRWD1', 'RBM5', 'CSTF2', 'FTO', 'SLTM'], ['GNL3', 'EWSR1', 'AARS', 'NKRF', 'FAM120A', 'PCBP2', 'FASTKD2', 'EFTUD2', 'PRPF8', 'RBM22', 'IGF2BP2', 'BUD13', 'SAFB', 'TIAL1', 'NOL12', 'IGF2BP1', 'PCBP1', 'SLBP', 'AQR', 'DDX55', 'APOBEC3C', 'G3BP1', 'KHDRBS1', 'PABPN1', 'TARDBP', 'HNRNPM', 'TIA1', 'YBX3', 'RPS5', 'TNRC6A']]

pmax = 2805

if moder == "unknown_shared_reduced_broad_share":
    filennum_per_group = 300
else:
    filennum_per_group = 1001

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


def rtokenize(vocab_list, seq):  # vocab_list = ["A", "G", "C", "U"]
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
    if int(label) == 1:
        new_label_list = np.array([[0, 1]], dtype="float32")
    else:
        new_label_list = np.array([[1, 0]], dtype="float32")
    return new_label_list


def main(args):  # gnum (gpu_id, protein_group_id)
    gnum = args[0]
    dirname = f"mydata_nored_f_{args[1]}_F_{args[2]}"
    output = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/{dirname}/"
    try:
        makedir_if_not_exist(output)
    except Exception as e:
        print(e)
    tape_dict = get_tape_out_dict()
    if gnum == 9:
        thisrange = targetrange[eachlen * gnum:]
    else:
        thisrange = targetrange[eachlen * gnum: eachlen * (gnum + 1)]
    for i in thisrange:
        try:
            arr = np.load(f"{input}/{i}.npy", allow_pickle=True)
            hbpot_arr = np.load(f"{hbpot_path}/{i}.npz", allow_pickle=True)["pot"]
            pipot_arr = np.load(f"{pipot_path}/{i}.npz", allow_pickle=True)["pot"]
        except Exception as e:
            print(e)
            continue
        pro_id_arr, pro_token_arr, rna_token_arr, label_arr, pro_mask_arr, cross_mask_arr, p_tape_arr, hpots, ppots = \
            None, None, None, None, None, None, None, None, None
        try:
            for idx, (item, hpot, ppot) in enumerate(zip(arr, hbpot_arr, pipot_arr)):
                # item = [proid, proseq, rseq, label, reducedproseq, reduce_index]
                pro_array_tokenized = ptokenize(PCODES, item[1], pmax)
                rna_array_tokenized = rtokenize(RCODES, item[2])
                pro_len = len(item[1])
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
                    try:
                        rna_token_arr = np.concatenate([rna_token_arr, np.array(rna_array_tokenized)[np.newaxis, :]], axis=0)
                    except ValueError:
                        print("error ")
                        print(f"rna_array_tokenized {rna_token_arr.shape}")
                        print(f"rna_array_tokenized to add {np.array(rna_array_tokenized)[np.newaxis, :].shape}")
                        continue
                    label_arr = np.concatenate([label_arr, make_label_vec(item[3])], axis=0)
                    pro_mask_arr = np.concatenate([pro_mask_arr, make_pmask(pro_len)], axis=0)
                    cross_mask_arr = np.concatenate([cross_mask_arr, make_cross_mask(pro_len)], axis=0)
                    hpots = np.concatenate([hpots, hpot[np.newaxis, :]], axis=0)
                    ppots = np.concatenate([ppots, ppot[np.newaxis, :]], axis=0)
        except TypeError:
            pass
        # # ogroup: "all5_small_tape" when known protein
        if WRITE == 1:
            np.savez_compressed(f"{output}/{i}", proid=pro_id_arr, protok=pro_token_arr, rnatok=rna_token_arr, p_tape_arr=p_tape_arr,
                label=label_arr, pro_mask=pro_mask_arr, cross_mask=cross_mask_arr, hb_pots=hpots, pi_pots=ppots)


def tardir(fval, large_fval):
    dirname = f"mydata_nored_f_{fval}_F_{large_fval}"
    os.chdir("/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/")
    # call([f"cd /Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/"], shell=True)
    call([f"tar -cvzf mydata_nored.tar.gz {dirname}"], shell=True)
    call([f"scp -r -P 3939 /Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/mydata_nored.tar.gz kimura.t@131.112.137.52:/home/kimura.t/rbpcamas/data/"], shell=True)


def runn8(fval, large_fval):
    p = Pool(10)
    arglist = []
    for i in range(10):
        arglist.append([i, fval, large_fval])
    p.map(main, arglist)
    p.close()
    tardir(fval, large_fval)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

if __name__ == '__main__':
    runn8()
