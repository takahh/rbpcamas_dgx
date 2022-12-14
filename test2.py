import os


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    path = "/Users/mac/Documents/RBP_CAMAS/data/newdata/base_bed_files/chr1/SFPQ.bed"
    prolist = []
    with open(f"{path}") as f:
        for lines in f.readlines():
            ele = lines.split()
            flength = int(ele[2]) - int(ele[1]) + 1
            if flength > 101:
                print(flength)
