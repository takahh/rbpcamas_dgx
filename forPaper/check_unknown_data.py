# -------------------------------------------------------------------
# this code confirms unknown data set is exclusive in each group
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import numpy as np
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/training_data_tokenized/"  # 0/0.npz

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------
group0 = []
group1_4 = []


def main():
    # collect proid for group 0 and group 1-4
    for i in range(5):
        for files in os.listdir(f"{path}{i}"):
            arr = np.load(f"{path}{i}/{files}")
            for proid in arr["proid"]:
                if i == 0:
                    if proid not in group0:
                        group0.append(proid.tolist()[0])
                else:
                    if proid not in group1_4:
                        group1_4.append(proid.tolist()[0])
    print(group0)
    print(group1_4)

    for item in group0:
        if item in group1_4:
            print(f"{item} in group1_4!!!")



# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()