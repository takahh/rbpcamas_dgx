# -------------------------------------------------------------------
# this code simply prints protein lengths
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
import os

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
proseq_master = f"/Users/mac/Documents/transformer_tape_dnabert/data/proteinseqs.csv"
tape = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/past/protein/"
problematic = ["AATF", "CSTF2", "SAFB", "QKI"]
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    with open(proseq_master) as f:
        for lines in f.readlines():
            ele = lines.strip().split(",")
            tape_len = np.load(f"{tape}{ele[0]}.npy")[0, -1, :4]
            if ele[0] in problematic:
                print(f"{ele[0]} -- {len(ele[1])}, tape {tape_len}")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()