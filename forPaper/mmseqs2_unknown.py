# -------------------------------------------------------------------
# this code run mmseqs2 for all fasta file of RBPsuite
# mmseqs easy-linclust /Users/mac/Downloads/train_dir/AARS.negative.fa clusterRes tmp
# input : /Users/mac/Downloads/train_dir/AARS.negative.fa
# output: /Users/mac/Documents/transformer_tape_dnabert/data/clusteredRBPsuite
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import numpy as np
from subprocess import call
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
bpath = f"{path}selected_npy_per_RNA/"
path_all_np = f"{bpath}final_all.npy"
path_all = f"{bpath}final_all.txt"
path_all_with_headers = f"{bpath}final_all_with_headers"
path_all_out = f"{bpath}final_all_representatives"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def change_numpy_to_text():
    arr = np.load(path_all_np)
    with open(path_all, "w") as f:
        for item in arr:
            f.writelines(f"{item}\n")


def make_ifile_with_headers():
    count = 0
    with open(f"{path_all}") as f, open(f"{path_all_with_headers}", "w") as fo:
        for lines in f.readlines():
            fo.writelines(f">entry_{count}\n{lines}")
            count += 1


def main():
    os.chdir(bpath)
    call([f"mmseqs easy-linclust {path_all_with_headers} {path_all_out} tmp"], shell=True)
    # call([f"rm -Rf ./tmp"], shell=True)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    change_numpy_to_text()
    make_ifile_with_headers()
    main()