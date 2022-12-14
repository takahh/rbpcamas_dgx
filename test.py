import os

import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


def main():
    path = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/overlaps/chr1/"
    for dirs in os.listdir(path):
        if "." in dirs:
            continue
        for files in os.listdir(f"{path}{dirs}"):
            if ".bed" not in files:
                continue
            with open(f"{path}{dirs}/{files}") as f:
                for lines in f.readlines():
                    ele = lines.split()
                    nagasa = int(ele[2]) - int(ele[1])
                    if int(nagasa) > 101:
                        print(dirs)
                        print(lines)


if __name__ == "__main__":
    main()
