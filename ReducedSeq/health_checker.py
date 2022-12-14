# -------------------------------------------------------------------
# this code checks validity of the tokeized data before running models
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
import os
from zipfile import BadZipFile

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
reducelevel = 9  # 13, 9, 6
input = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/training_data_tokenized/1400_shared_20/"
input = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/training_data_tokenized/shared_{reducelevel}/"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    count = 0
    for dir in os.listdir(input):
        count += 1
        # if dir not in ["1012.npz", "664.npz"]:
        #     continue
        if ".npz" not in dir:
            continue
        try:
            arr = np.load(f"{input}{dir}", allow_pickle=True)
            if len(arr.files) != 11:
                print(arr.files)
                print(f"Item is missing {dir}")
        except BadZipFile:
            print(f"badzip {dir}")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()