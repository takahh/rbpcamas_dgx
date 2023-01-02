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
bpath = "/home/kimura.t/rbpcamas/batch_files/Protein_centric/nored/noaug/"
figpath = f"/home/kimura.t/rbpcamas/python/Figures/protein_cent/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_data_per_file(filename):
    with open(filename) as f:
        allauclist, alllosslist = [], []
        print("working...")
        for lines in f.readlines():
            if "loss" in lines and "step" in lines:
                # 50/50 [==============================] - 164s 3s/step - loss: 0.6608 - auc: 0.6288
                loss = lines.split("loss:")[1].split("auc")[0].strip()
                print(f"###{loss}###")
        return alllosslist


def get_file_names_from_old_to_new():
    errfilelist = [f"{bpath}{dirs}" for dirs in os.listdir(bpath) if "out" in dirs]
    errfilelist.sort(key=lambda x: os.path.getmtime(x))
    print(errfilelist)
    return errfilelist


def main():
    alllosslist = []
    filelist = get_file_names_from_old_to_new()
    for filename in filelist:
        alllosslist.extend(get_data_per_file(filename))
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    plt.plot(range(len(alllosslist)), alllosslist)
    plt.savefig(f"{figpath}noaug.png")
    plt.show()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()
