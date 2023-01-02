# -------------------------------------------------------------------
# this code plots from raw log data
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------

import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
bpath = "/home/kimura.t/rbpcamas/batch_files/Protein_centric/nored/"
figpath = f"/home/kimura.t/rbpcamas/python/Figures/protein_cent/"
label_dict = {"noaug": "Not Augmented", "aug": "Augmented"}
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_data_per_file(filename):
    with open(filename) as f:
        auclist, losslist = [], []
        for lines in f.readlines():
            if "loss" in lines and "step" in lines:
                # 50/50 [==============================] - 164s 3s/step - loss: 0.6608 - auc: 0.6288
                loss = float(lines.split("loss: ")[1].split(" - auc")[0])
                losslist.append(loss)
        return losslist


def get_file_names_from_old_to_new(path):
    errfilelist = [f"{path}{dirs}" for dirs in os.listdir(path) if "out" in dirs]
    errfilelist.sort(key=lambda x: os.path.getmtime(x))
    return errfilelist


def main():
    plt.figure()
    for mode in ["aug", "noaug"]:
        alllosslist = []
        filelist = get_file_names_from_old_to_new(f"{bpath}{mode}/")
        for filename in filelist:
            alllosslist.extend(get_data_per_file(filename))
        plt.plot(range(len(alllosslist)), alllosslist, label=label_dict[mode])
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.savefig(f"{figpath}loss.png")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()
