# -------------------------------------------------------------------
# this code shuffles data from 1,1,1,....1,0,0,0 to 1,1,0,1,0,1...
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
import random

path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/benchmarks/label/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    for files in os.listdir(path):
        if "_pos_" not in files and ".md" not in files:
            ofile = f"{path}{files}".replace(".txt", "_shuffled.txt")
            with open(ofile, "w") as fo:
                with open(f"{path}{files}") as f:
                    line_list = f.read().split("\n")
                    random.shuffle(line_list)
                    for item in line_list:
                        if len(item) > 3:
                            fo.writelines(f"{item}\n")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()