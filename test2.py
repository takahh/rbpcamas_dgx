import numpy as np


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    path = "/Users/mac/Documents/RBP_CAMAS/data/newdata/batched_nparray"
    arr = np.load(path, allow_pickle=True)
    for files in arr.files:
        if files == "hb_pots":
            print(f"{files}: {arr[files]}")

    # path = "/Users/mac/Documents/RBP_CAMAS/data/newdata/attn_arrays_hb/all_8/2.npz"
    # arr = np.load(path, allow_pickle=True)
    # print(arr["pot"].shape)
    # print(arr["pot"])
