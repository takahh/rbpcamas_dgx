# -------------------------------------------------------------------
# this code takes information from logs

# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
from sklearn import metrics, roc_curve
from matplotlib import pyplot as plt
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
logpath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/single_node/tape_aug_no_clip.e10943009"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    hostlist = []
    with open(logpath) as f:
        for lines in f.readlines():
            if "ROC" in lines:  # ROC_DATA:r2i2n7:
                hostname = lines.split(":")[1]
                if hostname not in hostlist:
                    hostlist.append(hostname)
            if len(hostlist) == 32:
                break
    count = 0
    lossdict, aucdict = {}, {}
    pred_arr = None
    label_arr = None
    for item in hostlist:
        print(item)
        subauclist, sublosslist = [], []
        with open(logpath) as f:
            for lines in f.readlines():
                if "ROC" in lines and item in lines:
                    count += 1
                    listt, sublist = [], []
                    line1 = lines.split(":")[2].split("-")[0].strip("]").split("[")
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
                    list2 = [x.strip("[").strip("]").split(" ") for x in list2[1:-1].split("] [")]
                    list2 = [[int(y) for y in x] for x in list2]

                    if count == 1:
                        pred_arr = np.array(list1)
                        label_arr = np.array(list2)
                    else:
                        pred_arr = np.concatenate([pred_arr, np.array(list1)])
                        label_arr = np.concatenate([label_arr, np.array(list2)])
                    if count % 100 == 0:
                        subauclist.append(metrics.roc_auc_score(label_arr, pred_arr))
                        sublosslist.append(metrics.log_loss(label_arr, pred_arr))
                        count = 0
                        pred_arr = None
                        label_arr = None

        aucdict[item] = subauclist
        lossdict[item] = sublosslist
    print(aucdict)
    plt.figure()
    for host in hostlist:
        plt.scatter(range(len(aucdict[host])), aucdict[host])
    plt.show()
    plt.figure()
    for host in hostlist:
        plt.scatter(range(len(lossdict[host])), lossdict[host])
    plt.show()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()