# -------------------------------------------------------------------
# this code makes statistical potential tables for each pair of RNA-protein
# and store it in HDD
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
import pandas as pd
import sys
from scipy.special import softmax
import os

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------

# dict constant
aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
          'SER', 'THR', 'TRP', 'TYR', 'VAL']
baces = ['A', 'C', 'G', 'U']
aminos_pi = ["ARG", "TRP", "ASN", "HIS", "GLU", "GLN", "TYR", "PHE", "ASP"]
one2three_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
best_pot_normed = f"/Users/mac/Documents/T3_groupdisk_download_manual/RNPopt/RNPopt/data/result/eval4/optimized_normed_pot_list/best_pot_subset1_nocv.csv"
promaxdict = {20: 2805}
maxprolen = promaxdict[20]

# paths
path = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/"
inputpath = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/batched_nparray/nored/"


# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def make_if_not_exist(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def get_attn_dict(type):
    attn_list = pd.read_csv(best_pot_normed, header=None).values[0]
    attn_list = -attn_list
    # attn_list = softmax(-attn_list)
    attention_dict = {}
    if type == "PI":
        count = 80
        for amino in aminos_pi:
            for base in baces:
                attention_dict[f"{amino}_{base}"] = attn_list[count]
                count += 1
    else:
        count = 0
        for amino in aminos:
            for base in baces:
                attention_dict[f"{amino}_{base}"] = attn_list[count]
                count += 1
                if count == 80:
                    break
    return attention_dict


def get_seq_dict(path, start, maxfilenum):
    seq_dict = {}
    for i in range(start, maxfilenum):
        try:
            # arr = np.load(f"{path}/{i}.npy")
            arr = np.load(f"{path}/{i}.npy", allow_pickle=True)
        except (FileNotFoundError, OSError):
            continue
        # arr = np.load(f"{path}{group}/{i}.npy")
        seq_dict[i] = arr
    print("seq dict done")
    return seq_dict


def main(mode, type, maxfilenum, seq_file, startnum=0, group=None):  # mode : known or not, tyoe: hb / pi
    # load attn values
    attn_dict = get_attn_dict(type)
    # load all npy into a dictionary
    seq_dict = get_seq_dict(seq_file, startnum, maxfilenum)
    zero_row = [0] * 101

    # identify output path
    if mode == "known":
        if type == "HB":
            attn_out_dir = f"{path}nonreduced/attn_arrays_hb/"
        else:
            attn_out_dir = f"{path}nonreduced/attn_arrays_pi/"

    # for i in range(maxfilenum - 1, 0, -1):
    for i in range(startnum, maxfilenum):
        all_list = None
        for idx, item in enumerate(seq_dict[i]):  # repeat 5 times
            # print("|")
            # when unknown, item has 4 values (id, proseq, rnaseq, label)
            # when known,
            attn_to_write_list = []
            # # change here
            rna_seq = item[2]
            protein_seq = item[1]
            for pcount in range(maxprolen):
                # print(pcount)
                row_list = []
                if pcount < len(protein_seq):
                    try:
                        # in case of canonical residue
                        residue = one2three_dict[protein_seq[pcount]]
                    except KeyError:
                        # when uncanonical residue, add zero row and skip the rest
                        row_list = zero_row
                        attn_to_write_list.append(row_list)
                        continue
                    for rcount in range(101):
                        if rcount < len(rna_seq):
                            try:
                                row_list.append(attn_dict[f"{residue}_{rna_seq[rcount]}"])
                                # if type == "PI" and rcount < 5 and pcount < 10:
                                #     print(f"{idx}{pcount}{rcount} {residue}_{rna_seq[rcount]}")
                            except KeyError:
                                row_list.append(0)
                        else:
                            row_list.append(0)
                    attn_to_write_list.append(row_list)
                else:
                    row_list = [zero_row] * (maxprolen - len(protein_seq))
                    if all_list is None:
                        all_list = np.concatenate([attn_to_write_list, row_list])[np.newaxis, :, :]
                    else:
                        here_list = np.concatenate([attn_to_write_list, row_list])[np.newaxis, :, :]
                        all_list = np.concatenate([all_list, here_list], axis=0)
                    break
        np.savez_compressed(f"{attn_out_dir}/{i}", pot=np.array(all_list))


def runn7():
    os.environ['TMPDIR'] = "/Users/mac/Downloads/tmp"
    mode = "known"  # known protein patten
    for type in ["HB", "PI"]:
        maxfile_num = 1002
        start_num = 0
        seq_file = f"{inputpath}"
        main(mode, type, maxfile_num, seq_file, start_num)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    os.environ['TMPDIR'] = "/Users/mac/Downloads/tmp"
    runn7()
