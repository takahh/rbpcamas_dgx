# -------------------------------------------------------------------
# this code calculates auc from log npy
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Documents/LPI_BLS/LPI_BLS/results/"
label = f"{path}predicted_probs_1000_value.npy"
pred = f"{path}predicted_probs_1000_label.npy"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    arr_label = np.load(label, allow_pickle=True)
    arr_pred = np.load(pred, allow_pickle=True)
    auc = roc_auc_score(arr_pred, arr_label)
    print(auc)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()