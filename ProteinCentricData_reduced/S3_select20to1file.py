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
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def runs3(max_sites=1000):
    datalist = []
    with open(f"{opath}", "w") as fo:
        for files in os.listdir(path):
            if ".csv" in files:
                with open(f"{path}{files}") as f:
                    for lines in f.readlines():
                        # chr19-582939-583038,135,"['WDR43', 'DDX6',...]
                        freq = int(lines.split(",")[1])
                        # filter by positive count
                        # if least_posi_count_per_rna - 1 < freq < 150 - least_posi_count_per_rna:
                        datalist.append(f"{files},{lines}")
        shuffle(datalist)
        fo.writelines(datalist[:max_sites])


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    runs3()
