# -------------------------------------------------------------------
# this code plots stat data of per RNA data generated by
# /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/python/forPaper/analyze_RBPsuite_per_RNA.py
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os, time, datetime
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
bpath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
ipath = f"{bpath}stats_of_bind_data_per_RNA/"
odir = f"{bpath}selected_npy_per_RNA/"
opath1 = f"{odir}selected_RNAs"  # >= 2 proteins
opath2 = f"{odir}must_RNAs"  # >= 3 proteins

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def freq_count(array):
    print(array)
    values, counts = np.unique(array, axis=0, return_counts=True)
    with_freq_arr = [[x.tolist()[0], x.tolist()[1], y] for x, y in zip(values, counts)]
    return with_freq_arr


def main():
    must_data_count = 0
    rest_data_count = 0
    plot_data_arr = None
    selected_file_list = []
    must_list = []
    for dirs in sorted(os.listdir(ipath)):
        path = f"{ipath}{dirs}"
        for files in os.listdir(path):
            arr = np.load(f"{path}/{files}", allow_pickle=True)  # count of 1, count of 0
            unique, counte = np.unique(arr, return_counts=True)
            countdict = dict(zip(unique, counte))
            try:
                onecount = int(countdict[0])
                zerocount = int(countdict[1])
            except KeyError:
                continue
            if onecount * zerocount != 0:
                positive_fraction = onecount/(onecount + zerocount)
                if 0.3 <= positive_fraction <= 0.7:
                    if onecount * zerocount > 1:
                        must_data_count += onecount + zerocount
                        print(f"must {must_data_count}, rest {rest_data_count}")
                        must_list.append([files, onecount + zerocount, positive_fraction])
                    else:
                        pass
                        # # write selected arr to the different directory
                        # rest_data_count += onecount + zerocount
                        # print(f"must {must_data_count}, rest {rest_data_count}")
                        # selected_file_list.append(files)
                        # if plot_data_arr is None:
                        #     plot_data_arr = np.array([[onecount, zerocount]])
                        # else:
                        #     plot_data_arr = np.append(plot_data_arr,  np.array([[onecount, zerocount]]), axis=0)
            # if data_count > 20000:
            #     break
    # np.save(opath1, selected_file_list)
    np.save(opath2, must_list)
    # plot_data_arr = freq_count(plot_data_arr)
    # # print(plot_data_arr)
    # plt.figure()
    # xarr = [x[0] for x in plot_data_arr]
    # yarr = [x[1] for x in plot_data_arr]
    # zarr = [x[2] for x in plot_data_arr]
    # plt.scatter(xarr, yarr, c=zarr, s=50)
    # plt.show()
    # plt.figure()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()