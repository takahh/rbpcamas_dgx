# -------------------------------------------------------------------
# this code moves tokenized 1400 files after tokenizer
reduce_level = 13
# input = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/training_data_tokenized/shared_{reduce_level}/"
input = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/training_data_tokenized_broad_share_red_8/"
output = f"{input}1200/"
# output = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/training_data_tokenized/1400_shared_{reduce_level}/"
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os, shutil
from random import shuffle
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------

if not os.path.exists(output):
    os.mkdir(output)


def main():

    # -------------------------
    # collect all data paths
    # -------------------------
    datalist = []
    for i in range(5):
        for dirs in os.listdir(f"{input}{i}/"):
            if "npz" not in dirs:
                continue
            filesize = os.path.getsize(f"{input}{i}/{dirs}")
            if filesize > 3600:
                # collect
                datalist.append(f"{input}{i}/{dirs}")
    print(f"{len(datalist)} is the total file count")
    # -----------------------
    # shuffle data and cope to the target dir
    # -----------------------
    shuffle(datalist)
    count = 0
    for item in datalist:
        shutil.copy2(item, f"{output}{count}.npz")
        count += 1


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()