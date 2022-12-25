# -------------------------------------------------------------------
# this code coolects rna fragmenta that are shared by 20-25 proteins
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
from random import shuffle
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Documents/RBP_CAMAS/data/newdata/counts/"
opath = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq20_25.csv"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    datalist = []
    with open(f"{opath}", "w") as fo:
        for files in os.listdir(path):
            if ".csv" in files:
                with open(f"{path}{files}") as f:
                    for lines in f.readlines():
                        freq = int(lines.split(",")[1])
                        if 19 < freq < 26:
                            datalist.append(f"{files},{lines}")
        shuffle(datalist)
        fo.writelines(datalist[:200])


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()