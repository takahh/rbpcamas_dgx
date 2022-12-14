# making mask data for RNA for benchmarks

import numpy as np
from matplotlib import pyplot as plt


def main():
    path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/past/proteinseqs.csv"
    for maxnum in [3680, 2805, 1510]:
        pronamelist, groupedlist = [], []
        with open(path) as f:
            for lines in f.readlines():
                if len(lines.strip().split(",")[1]) < maxnum:
                    pronamelist.append(lines.strip().split(",")[0])
        for i in range(5):
            if i != 4:
                groupedlist.append(pronamelist[30 * i: 30 * (i+1)])
            else:
                groupedlist.append(pronamelist[30 * i:])

        print(groupedlist)

if __name__ == "__main__":
    main()