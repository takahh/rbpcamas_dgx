# -------------------------------------------------------------------
# this code makes a "TEST SET" list of id for each fold in each benchmark
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------
import os
import random
import numpy as np


def main():
    idlist = []
    arr_dict = {}
    path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/benchmarks/label/"
    outfile = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/benchmarks/id_list_for_5CV.csv"
    for file in os.listdir(path):
        if "pair" in file and "pos" not in file:
            benchname = file.split("_")[0]
            count = 0
            # count the number of data in the benchmark set
            with open(f"{path}{file}") as f:
                for lines in f.readlines():
                    count += 1
            id_list = list(np.arange(count))
            # shuffle #################
            random.shuffle(id_list)
            ###########################
            fold_size = round(len(id_list)/5) - 1
            for i in range(5):
                if i != 4:
                    idlist.append(id_list[i * fold_size: (i + 1) * fold_size])
                else:
                    idlist.append(id_list[i * fold_size:])
            arr_dict[benchname] = idlist
            np.savez_compressed(outfile.replace("id_list", f"id_list_{benchname}"), list=idlist)
            idlist = []
            arr_dict = {}


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()