# -------------------------------------------------------------------
# this code select 6000 sequences from reps
# take the high count ones up to count 836 and randomely take the rest
# so that the total count is 6,000
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os.path
from random import shuffle
import numpy as np
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/selected_npy_per_RNA/"
ipath = f"{path}final_all_representatives_rep_seq.fasta"
opath = f"{path}final_6000.npy"
bind_data_path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/bind_data_per_RNA/"
outpath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/selected_npy_per_RNA/inputdata/"
posinegadict = {"1":"positive", "0":"negative"}
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def collect_high_count():
    take_the_next = 0
    selected_list = []
    rest_list = []
    high_count = 0
    with open(ipath) as f:
        for lines in f.readlines():
            if ">" in lines:
                if int(lines.split("_")[1].strip()) < 836:
                    take_the_next = 1
            else:
                if take_the_next == 1:
                    selected_list.append(lines.strip())
                    high_count += 1
                    take_the_next = 0
                else:
                    rest_list.append(lines.strip())

    return selected_list, rest_list


def add_one_rna_line(protein_name, label, rna_sequence):
    odir = f"{outpath}/"
    filename = f"{protein_name}{posinegadict[label]}fa_rep_seq.fasta"
    if not os.path.exists(f"{odir}/{filename}"):  # file not exist
        with open(f"{odir}/{filename}", "w") as f:  # DDX51positivefa_rep_seq.fasta.npy
            f.writelines(rna_sequence + "\n")
    else:                                     # file exists
        with open(f"{odir}/{filename}", "a") as f:  # DDX51positivefa_rep_seq.fasta.npy
            f.writelines(rna_sequence + "\n")


def get_label_and_pro(rnaseq):  # /AAAAA/
    with open(f"{bind_data_path}{rnaseq[:5]}/{rnaseq}") as f:
        for lines in f.readlines():
            # AARS,1
            ele = lines.strip().split(",")
            add_one_rna_line(ele[0], ele[1], f"{rnaseq}")


def add_label_protein(rnalist):
    label_added_list = []
    count = 0
    for rnaseq in rnalist:
        count += 1
        print(count)
        label_added_list.append(get_label_and_pro(rnaseq))


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # get high count data
    high_count_list, remaining_list = collect_high_count()
    samplenum_from_remain = 17000 - len(high_count_list)
    shuffle(remaining_list)

    # get remaining data
    to_add_list = remaining_list[:samplenum_from_remain]

    # combine
    high_count_list.extend(to_add_list)

    # add label and protein info
    add_label_protein(high_count_list)
