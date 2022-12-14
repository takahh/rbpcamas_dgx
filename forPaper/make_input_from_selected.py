# -------------------------------------------------------------------
# this code selects data from selected rnas and prepare input data
# for unknown protein calculation
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
from plot.plot_stat_of_data_per_RNA import freq_count
from matplotlib import pyplot as plt
from random import shuffle

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
bpath = f"{path}selected_npy_per_RNA/"
must_path = f"{bpath}must_RNAs.npy"  # >= 3 結合データ以上
# rest_path = f"{bpath}selected_RNAs.npy"  # >= 2 結合データ
datapath = f"{path}bind_data_per_RNA/"

final_opath_all = f"{bpath}final_all"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def plot_must_data(target_path):
    arr = np.load(target_path, allow_pickle=True)
    unique, count = np.unique(arr[:, 1:], return_counts=True, axis=0)
    arr2 = []
    for idx, item in enumerate(unique):
        arr2.append([float(item[0]), float(item[1]), count[idx]])
    arr2 = np.array(arr2)
    z = arr2[:, 0]
    y = arr2[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(z, y)
    for i, txt in enumerate(arr2[:, 2]):
        ax.annotate(txt, (z[i], y[i]))
    plt.xlabel("Protein Count")
    plt.ylabel("Positive Fraction")
    plt.show()


def select_data(target_path):
    # separate high count > 3 anf the rest
    arr = np.load(target_path, allow_pickle=True)
    must_arr = [list(x)[0][:-4] for x in arr if x[1].astype("int") > 3]  # this is fixed
    print(len(must_arr))
    rest_arr = [list(x)[0][:-4] for x in arr if x[1].astype("int") <= 3]
    must_arr.extend(rest_arr)
    np.save(final_opath_all, must_arr)


def main():
    # plot_must_data(must_path)
    select_data(must_path)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()