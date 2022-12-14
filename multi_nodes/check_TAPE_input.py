# -------------------------------------------------------------------
# this code checks input data with TAPE features
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/training_data_tokenized/all5_small_TAPE/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    for files in os.listdir(path):
        # if "0.npz" != files:
        #     continue
        arr = np.load(f"{path}{files}", allow_pickle=True)
        for name in arr.files:
            subarr = arr[name]
            print(np.isnan(np.min(subarr)))


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()