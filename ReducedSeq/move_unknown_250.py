# -------------------------------------------------------------------
# this code moves 250 tokenized data files per unknown group
import os

input = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/training_data_tokenized_broad_share/"  # 4_old
output = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/training_data_tokenized_broad_share/"  # 4
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import shutil
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    for i in range(5):
        input = f"{input}{i}_old/"
        output = f"{input}{i}/"
        if not os.path.exists(output):
            os.mkdir(output)

        for dirs in os.listdir(input):
            if "npz" not in dirs:
                continue
            if int(dirs.replace(".npz", "")) < 250:
                shutil.move(f"{input}{dirs}", f"{output}{dirs}")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()