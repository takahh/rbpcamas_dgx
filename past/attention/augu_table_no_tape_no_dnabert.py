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
from scipy.special import softmax

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# basepath = "/Users/mac/Desktop/t3_mnt/"
basepath = "/gs/hs0/tga-science/kimura/"
aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
          'SER', 'THR', 'TRP', 'TYR', 'VAL']
baces = ['A', 'C', 'G', 'U']
aminos_pi = ["ARG", "TRP", "ASN", "HIS", "GLU", "GLN", "TYR", "PHE", "ASP"]
one2three_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
best_pot_normed = f"{basepath}RNPopt/data/result/eval4/optimized_normed_pot_list/best_pot_subset1_nocv.csv"

benchmark_set = "RPI369"
path = f"{basepath}transformer_tape_dnabert/data/"
attn_input_pairs = f"{path}benchmarks/label/{benchmark_set}_pairs_shuffled.txt"
RNA_seq_file = f"{path}benchmarks/sequence/{benchmark_set}_rna_seq.fa"
protein_seq_file = f"{path}benchmarks/sequence/{benchmark_set}_protein_seq.fa"
# attn_out_dir = f"{path}attn_arrays_no_tape_no_dnabert/{benchmark_set}/"
attn_out_dir = f"{path}attn_arrays_pi_no_tape_no_dnabert/{benchmark_set}/"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_attn_dict():
    attn_list = pd.read_csv(best_pot_normed, header=None).values[0]
    attn_list = softmax(-attn_list)
    attention_dict = {}
    count = 80
    # count = 0
    # for amino in aminos:
    for amino in aminos_pi:
        for base in baces:
            attention_dict[f"{amino}_{base}"] = attn_list[count]
            count += 1
            # if count == 80:
            #     break
    return attention_dict


def get_seq_dict(path):
    seq_dict = {}
    with open(path) as f:
        data_list = f.read()
        data_list = data_list.replace("\n>", ",").replace("\n", ":").replace(">", "").split(",")
        for item in data_list:
            ele = item.split(":")
            seq_dict[ele[0]] = ele[1]
    return seq_dict


def main():
    attn_dict = get_attn_dict()
    RNA_seq_dict = get_seq_dict(RNA_seq_file)
    protein_dict = get_seq_dict(protein_seq_file)
    zero_row = [0] * 4001

    # make attention table
    with open(attn_input_pairs) as f:
        # per pair, make one table
        for lines in f.readlines():
            attn_to_write_list = []
            ele = lines.split()
            rna_seq = RNA_seq_dict[ele[1]]
            protein_seq = protein_dict[ele[0]]
            # for residue in protein_seq:
            for pcount in range(3680):
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
                    for rcount in range(4000):
                        if rcount < len(rna_seq):
                            try:
                                row_list.append(attn_dict[f"{residue}_{rna_seq[rcount]}"])
                            except KeyError:
                                row_list.append(0)
                        else:
                            row_list.append(0)
                else:
                    row_list = zero_row
                attn_to_write_list.append(row_list)
            np.savez_compressed(f"{attn_out_dir}{ele[0]}_{ele[1]}", data=attn_to_write_list)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()