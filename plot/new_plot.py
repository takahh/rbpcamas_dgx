# -------------------------------------------------------------------
# this code plot loss and auc that TF printed
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import numpy as np
from matplotlib import pyplot as plt

# plt.rcParams["figure.figsize"] = (40, 160)
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
datalist = [
            ["Red8 Aug Mul Protein-centric",
                "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/mydata/pre_aug_mul/",
                "r8_uk2kn_pre_no_aug_mul.o"],
            ["Red8 Aug add Protein-centric",
             "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/mydata/pre_aug/",
             "mydata_r8_add.o"],
            ["Red8 No Aug Protein-centric",
                "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/mydata/pre_no_aug/",
                "r8_uk2kn_pre_no_aug.o"]
        ]
datalist2 = [
            ["Red8 Aug Mul Mixed",
                "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/new_old_mix/pre_aug_mul/",
                "new_old_mix_mul.o"],
            ["Red8 Aug add Mixed",
             "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/new_old_mix/pre_aug/",
             "mix_r8_add.o"],
            ["Red8 No Aug Mixed",
                "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/new_old_mix/pre_no_aug/",
                "new_old_mix_no_aug.o"]
        ]
datalist3 = [
            ["Red8 Aug add RNA centric",
             "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/RNAcentric/pre_aug/",
             "mix_r8_add.o"],
            ["Red8 No Aug RNA centric",
                "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/RNAcentric/pre_no_aug/",
                "RNAcentr_noaug.o"]
        ]
titlelist = ["Train Loss", "Train AUC", "Test AUC", "Test Loss"]
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_data_per_file(fpath):
    data_list = [[], [], [], []]
    with open(fpath) as f:
        #  "========] - 513s 13s/step - loss: 0.6832 - auc: 0.5823 - val_test_auc: 0.5589 - val_test_loss: 0.6931"
        datalines = [x for x in f.readlines() if "val_test_auc" in x]
        for i in range(4):
            data_list[i] = [float(x.split("-")[i + 2].split(":")[1].strip()) for x in datalines]
        return data_list


def main(dlist):
    # for i in range(4):
    fig = plt.figure(figsize=(4, 15))  # plt.figure(figsize=(1,1))
    axes = fig.subplots(4)
    # for i in range(4):
    #     axes[i].

    for items in dlist:  # per task
        alllist = [[], [], [], []]
        # --------------------------
        # get file names and dates
        # --------------------------
        efilelist = [filename for filename in os.listdir(items[1]) if items[2] in filename]
        datelist = [int(x.split(items[2])[1]) for x in os.listdir(items[1]) if items[2] in x]
        datelist.sort()
        # --------------------------
        # read the files and get data
        # --------------------------
        data_four = []
        for dates in datelist:
            file_to_open = [x for x in efilelist if str(dates) in x][0]
            data_four = get_data_per_file(f"{items[1]}{file_to_open}")
            # --------------------------
            # add four lists to the alllist
            # --------------------------
            for i in range(4):
                alllist[i].extend(data_four[i])
        # --------------------------
        # plot
        # --------------------------
        for i in range(4):  # train_loss, train_auc, test_loss, test_auc
            axes[i].set_xlim([0, 150])
            axes[i].set_ylim([0.4, 0.7])
            axes[i].plot(np.arange(len(alllist[i])), alllist[i], label=items[0])
            axes[i].set_title(titlelist[i])
            axes[i].legend()
    # for i in range(4):
    fig.show()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main(datalist)
    main(datalist2)
    main(datalist3)
