# -------------------------------------------------------------------
# this code collects rna fragmenta that are shared by 20-25 proteins
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
opath = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq25_32.csv"
MAX_SITE_COUNT = 200
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
                        if 25 < freq < 110:
                            datalist.append(f"{files},{lines}")
        shuffle(datalist)
        fo.writelines(datalist[:MAX_SITE_COUNT])


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()
