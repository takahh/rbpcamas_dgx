import os

import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


def main():
    path = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/mydata_nored_10_10/"
    for dirs in os.listdir(path):
        arr = np.load(f"{path}{dirs}")
        print(arr["label"])


if __name__ == "__main__":
    main()
