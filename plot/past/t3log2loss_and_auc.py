# -------------------------------------------------------------------
# this code reads log like no_tape_aug.e11091669 and calculates loss
# and auroc average
# input : /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/single_node/no_tape_aug/no_tape_aug.e11091669
# output: /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/log_summary/no_tape_aug.txt
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import numpy as np
from sklearn import metrics
import tensorflow as tf
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------

testauclist, testlosslist = [], []
trainauclist, trainlosslist = [], []
mode = "train"


def save_auc_and_loss(auc, loss_avg):  # save_auc_and_loss(auc, loss_avg)
    global mode, trainlosslist, trainauclist, testlosslist, testauclist
    # save test results
    if mode == "train":
        trainauclist.append(auc.result().numpy())
        trainlosslist.append(loss_avg.result().numpy())
        print(loss_avg.result().numpy())
        # print(auc.result().numpy())
        mode = "test"
    elif mode == "test":
        testauclist.append(auc.result().numpy())
        testlosslist.append(loss_avg.result().numpy())
        mode = "train"


def calc_auc_and_loss(lines, auc, loss_avg):
    global mode, trainlosslist, trainauclist, testlosslist, testauclist
    pred = cleaning([x for x in lines.split("::")[1].split("-tf.Tensor(")[0].split("] [")])
    label = cleaning([x for x in lines.split("-tf.Tensor(")[1].split(",")[0].split("]  [")])
    losses = tf.keras.metrics.binary_crossentropy(label, pred)
    loss = tf.reduce_mean(losses)
    auc.update_state(label, pred)
    loss_avg.update_state(loss)
    return auc, loss_avg


def cleaning(dirtylist):
    cleanedlist = [item.replace("[", "").replace("]", "") for item in dirtylist]
    cleanedlist = [x.split() for x in cleanedlist]
    cleanedlist = np.array(cleanedlist).astype(float)
    return cleanedlist


def get_loss_list(path, keywd):  # 1 batch 4 GPU, 250 batch = 1 epoch
    global mode, trainlosslist, trainauclist, testlosslist, testauclist
    auc = tf.keras.metrics.AUC()
    loss_avg = tf.keras.metrics.Mean()
    with open(path) as f:
        for lines in f.readlines():
            if "test ROC_DATA" in lines:
                auc, loss_avg = calc_auc_and_loss(lines, auc, loss_avg)
        print(keywd)
        print(loss_avg.result())
        print(auc.result())


def main():
    pathbase = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/test_single_node/"
    for dir in os.listdir(pathbase):
        for files in os.listdir(f"{pathbase}{dir}"):
            if ".e" in files:
                filepath = f"{pathbase}{dir}/{files}"
                keyword = filepath.split(".e")[0]
                get_loss_list(filepath, keyword)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()