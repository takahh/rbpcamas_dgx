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
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_data_per_file(filename):
    with open(f"{bpath}{filename}") as f:
        filenum = 200  # 2000/40 * 4
        allauclist, alllosslist = [], []
        subauclist_all, sublosslist_all, pred_arr, label_arr = [], [], None, None
        totalstep = 0
        epochcount = 0
        print("working...")
        for lines in f.readlines():
            if "ROC" in lines:
                totalstep += 1
                listt, sublist = [], []
                line1 = lines.split(":")[2].split("]-")[0].strip("]").split("[")
                list1 = [x.strip("'").strip(" ").strip("]").replace("  ", " ") for x in line1][2:]
                list1 = [x[:-1] if x[-1] == " " else x for x in list1]
                list1 = [x.split(" ") for x in list1]
                for x in list1:
                    for y in x:
                        if len(y) > 0:
                            sublist.append(y)
                    listt.append(sublist)
                    sublist = []
                list1 = [[float(e) for e in x] for x in listt]
                line2 = lines.split(":")[2].split("or(")[1]
                list2 = line2.split(",")[0]
                list2 = [x.strip("  [").split(" ") for x in list2.strip("[[[").split("]") if x]
                list2 = [[int(x) for x in item] for item in list2]
                # make two arrays for AUC caculation
                if totalstep == 1:
                    pred_arr = np.array(list1)
                    label_arr = np.array(list2)
                else:
                    pred_arr = np.concatenate([pred_arr, np.array(list1)])
                    label_arr = np.concatenate([label_arr, np.array(list2)])
                # calc. AUC and save to lists, the end of epoch
                if totalstep == filenum:
                    allauclist.append(metrics.roc_auc_score(label_arr[:, 1], pred_arr[:, 1]))
                    alllosslist.append(metrics.log_loss(  , pred_arr))
                    subauclist, sublosslist = [], []
                    totalstep = 0
                    epochcount += 1
        return alllosslist


def get_file_names_from_old_to_new():
    errfilelist = [dir for dir in os.listdir(bpath) if "err" in dir]
    errfilelist.sort(key=lambda x: os.path.getmtime(x))
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
