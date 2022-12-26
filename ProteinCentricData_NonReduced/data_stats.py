import os
import numpy as np
import sys

DIR = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/mydata_nored_15_15"


def main():
    proid_list, rtok_list = [], []
    negacount, posicount = 0, 0
    for files in os.listdir(DIR):
        print(f"{DIR}/{files}")
        arr = np.load(f"{DIR}/{files}")
        for label in arr["label"]:
            if label[0] == 0:
                negacount += 1
            else:
                posicount += 1
        for proid in arr["proid"]:
            proid_list.append(proid[0])
        for rnatok in arr["rnatok"]:
            rnastr = "".join(map(str, rnatok.tolist()))
            rtok_list.append(rnastr)
    unique_proteins = len(set(proid_list))
    unique_rnas = len(set(rtok_list))
    perRNA = (negacount + posicount)/unique_rnas
    perPRo = (negacount + posicount)/unique_proteins

    print(f"unique protein : {unique_proteins}")
    print(f"unique RNA : {unique_rnas}")
    print(f"posicount {posicount}")
    print(f"negacount {negacount}")
    print(f"perPRO {perPRo}")
    print(f"perRNA {perRNA}")


if __name__ == '__main__':
    main()