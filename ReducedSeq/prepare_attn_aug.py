# -------------------------------------------------------------------
# this code makes statistical potential tables for each pair of RNA-protein
# and store it in HDD
# input :
# output:
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

###### ####### ####### ####### ####### ####### ####### ####### #######
rmode = "shared_unknown_broad_share"  # all / lnc / lnc_unknown / attn_ana_hb / attn_ana_pi
REDUCE_LEVEL = 8  # 20, 13, 8, 4
PDBID = "6V5B_C_D"
####### ####### ####### ###### ####### ####### ####### ####### #######

basepath = "/Users/mac/Desktop/t3_mnt/"
# basepath = "/gs/hs0/tga-science/kimura/"
aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
          'SER', 'THR', 'TRP', 'TYR', 'VAL']
baces = ['A', 'C', 'G', 'U']
aminos_pi = ["ARG", "TRP", "ASN", "HIS", "GLU", "GLN", "TYR", "PHE", "ASP"]
one2three_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
best_pot_normed = f"/Users/mac/Documents/T3_groupdisk_download_manual/RNPopt/RNPopt/data/result/eval4/optimized_normed_pot_list/best_pot_subset1_nocv.csv"

benchmark_set = "RPI369"
path = f"{basepath}reduced_RBP_camas/data/"
# attn_input_pairs = f"{path}benchmarks/label/{benchmark_set}_pairs_shuffled.txt"
if rmode == "all":
    PATH2 = f"{path}known_protein_reduced/all_{REDUCE_LEVEL}/"
elif rmode == "shared":
    PATH2 = f"{path}known_protein_reduced/shared_{REDUCE_LEVEL}/"
    print(PATH2)
elif rmode == "shared_unknown":
    PATH2 = f"{path}/unknown_reduced/"
elif rmode == "attn_ana_hb":
    PATH2 = f"{path}/training_data/attn_analysis_hb"
elif rmode == "attn_ana_pi":
    PATH2 = f"{path}/training_data/attn_analysis_pi"
elif rmode == "shared_unknown_broad_share":
    path = f"{basepath}reduced_RBP_camas/data/"
    PATH2 = f"{path}/unknown_reduced_broad_share/"


promaxdict = {20: 2805, 13: 1609, 8: 986, 4: 609}
maxprolen = promaxdict[REDUCE_LEVEL]


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
    if mode == "unknown":
        if rmode == "shared_unknown_broad_share":
            keyword = "_broad_share"
        else:
            keyword = ""
        if type == "HB":
            attn_out_dir = f"{path}attn_arrays_hb{keyword}/{group}/"
        else:
            attn_out_dir = f"{path}attn_arrays_pi{keyword}/{group}/"
        make_if_not_exist(f"{path}attn_arrays_hb{keyword}/")
        make_if_not_exist(f"{path}attn_arrays_pi{keyword}/")

    elif mode == "known":
        if type == "HB":
            attn_out_dir = f"{path}attn_arrays_hb/{rmode}_{REDUCE_LEVEL}/"
        else:
            attn_out_dir = f"{path}attn_arrays_pi/{rmode}_{REDUCE_LEVEL}/"
    elif mode == "test":
        if type == "HB":
            attn_out_dir = f"{path}attn_arrays_hb/attn_ana/"
        else:
            attn_out_dir = f"{path}attn_arrays_pi/attn_ana/"

    make_if_not_exist(attn_out_dir)

    # for i in range(maxfilenum - 1, 0, -1):
    for i in range(startnum, maxfilenum):
        # if os.path.exists(f"{attn_out_dir}/{i}.npz"):
        #     print("exist!")
        #     continue
        all_list = None
        for idx, item in enumerate(seq_dict[i]):  # repeat 5 times
            # print("|")
            # when unknown, item has 4 values (id, proseq, rnaseq, label)
            # when known,
            attn_to_write_list = []
            # # change here
            # # protein seq reduced is item[4]
            rna_seq = item[2]
            protein_seq = item[4]
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
        # if type == "PI":
        #     print(all_list[4, 8, 0:10])
        # try:
        #     # np.savez(f"{attn_out_dir}/{i}", pot=np.array(all_list))
        #     print(np.array(all_list).shape)

        # print(attn_out_dir)
        np.savez_compressed(f"{attn_out_dir}/{i}", pot=np.array(all_list))

        #     print(f"skipping {i}...")
        # except OSError:
        #     pass


def test():
    attn_out_dir = f"{path}attn_arrays_hb/attn_ana/0.npz"
    arr = np.load(attn_out_dir, allow_pickle=True)
    print(arr["pot"])


def makeall():
    mode = "known"  # known protein patten
    for type in ["HB", "PI"]:
    # for type in ["PI"]:
        # type = sys.argv[1]
        maxfile_num = 3572
        start_num = 0
        seq_file = f"{PATH2}"
        main(mode, type, maxfile_num, seq_file, start_num)


def make_test():
    mode = "test"
    for type in ["PI", "HB"]:
        maxfile_num = 1
        seq_file = f"{PATH2}/"
        main(mode, type, maxfile_num, seq_file, 0)


def makefive():
    mode = "unknown"
    for group in range(5):
        for type in ["PI", "HB"]:
            if rmode == "shared_unknown_broad_share":
                maxfile_num = 600
            else:
                maxfile_num = 3572
            seq_file = f"{PATH2}{group}/"
            main(mode, type, maxfile_num, seq_file, 0, group)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    os.environ['TMPDIR'] = "/Users/mac/Downloads/tmp"
    # known proteins
    # makeall()
    # unknown proteins
    makefive()
    # make_test()
    # test()
