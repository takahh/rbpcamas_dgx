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
path = "/home/kimura.t/rbpcamas/batch_files/Protein_centric/nored/mydata_nored_15_152022-12-26-16-43_aug_1err.log"
figpath = f"{path.split('/')[-1]}plot.png"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    with open(path) as f:
        filenum = 160  # 4000 / 25 (20 / 4 * 5)
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
                    alllosslist.append(metrics.log_loss(label_arr, pred_arr))
                    subauclist, sublosslist = [], []
                    totalstep = 0
                    epochcount += 1
        plt.plot(range(len(alllosslist)), alllosslist)
        plt.savefig(figpath)
        plt.show()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()
