# -------------------------------------------------------------------
# this code summarize the result of
# /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/python/New_Dataset/find_same_RNA_seqs.py
# and save the stat as np array
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
import os
import numpy as np

bpath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
path = f"{bpath}bind_data_per_RNA/"
opath = f"{bpath}stats_of_bind_data_per_RNA/"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    for dirs in sorted(os.listdir(path), reverse=True):  # dirs="AAAAC"
        out_dir = f"{opath}{dirs[0:4]}"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        dirlist = os.listdir(out_dir)
        print(dirs)
        for files in os.listdir(f"{path}{dirs}"):
            out_path = f"{opath}{dirs[0:4]}/{files}"
            if f"{files}.npy" in dirlist:
                continue
            print(f"{files} working...")
            with open(f"{path}{dirs}/{files}") as f1:
                lines = f1.readlines()
                # SDAD1,1
                # ZNF800,1
                label_list = [int(x.split(",")[1].strip()) for x in lines]
            np.save(out_path, label_list)


# -----------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()