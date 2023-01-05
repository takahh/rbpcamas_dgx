# -------------------------------------------------------------------
# this code reads tblogs and make a plot of optimization
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------

exp = 3
experiment_id = "pMdW7AlIQFKT27BMuAPPYg"

# TARGETS = ["epoch_test_auc"]
TARGETS = ["epoch_test_auc", "epoch_loss", "epoch_test_loss"]
csv_path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/result/csv_from_tbdev/downloaded_data.csv"
kw_list_master = ["tape_aug_clip", "tape_no_aug", "tape_aug_only_pro", "tape_aug_no_2dsoft", "tape_aug_no_clip_augmultiply",
               "tape_aug", "no_tape_no_aug", "no_tape_aug", "tape_aug_only_rna", "lnc_tape_aug", "lnc_unknown_tape_aug"]


linestyle_list = ["-", ":", "-", "-", ":", (0, (1, 1)), (0, (5, 10)), (0, (5, 5)),
                  (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]
linewidth_list = [3.5, 1.0, 1, 2, 1.5]
color_list = ["black", "black", "black", "black", "black"]

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def download_csv_data_from_dev():
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df.to_csv(csv_path, index=False)


def plot_from_csv(target_value, expnum):
    fig_path = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/result/figs/{target_value}_{expnum}.png"
    # load dcsv file
    df = pd.read_csv(csv_path)
    count = 0
    plt.figure()
    fig, ax = plt.subplots(figsize=(11, 7))
    # fig, ax = plt.subplots()
    plt.tick_params(labelsize=16)
    plt.xlabel("Epochs", fontsize=22)
    plt.grid()

    if "test" not in target_value:
        plt.ylabel("Training Loss (Cross Entropy)", fontsize=20)
        # plt.xlim(0, 360)
        # plt.ylim(0.25, 0.70)
    else:  # test
        if "auc" in target_value:
            plt.ylabel("Test AUROC", fontsize=20)
            # plt.xlim(0, 360)
            # if expnum != 3:
            #     plt.ylim(0.66, 0.95)
        else:
            plt.ylabel("Test Loss (Cross Entropy)", fontsize=20)


    for keyword in kw_list:
        # select one series such as "tape_aug"
        subdf = df[df["run"].str.contains(f"keywrd_{keyword}_clip_coeff")]
        print(f"{keyword} -- {len(subdf)}")
        # select one value such from 'epoch_auc' 'epoch_loss' 'epoch_test_auc' 'epoch_test_loss'
        subdf = subdf[subdf["tag"] == target_value]
        print(subdf["run"].unique())
        # select train or test
        if "test" in target_value:
            subdf2 = subdf[subdf["run"].str.contains(f"validation")]
        else:
            subdf2 = subdf[subdf["run"].str.contains(f"train")]

        # add "date" column
        subdf2["date"] = subdf2["run"].apply(lambda x: x.split("t3_")[1].split("/")[0])

        # sort by the date
        sorted_df = subdf2.sort_values(by=["date"])

        # get values for plot
        steps = len(sorted_df["value"])
        if "auc" in target_value:
            plt.hlines(y=0.95, xmin=0, xmax=380)
        ax.plot(range(steps), sorted_df["value"], label=label_list[count], linestyle=linestyle_list[count],
                linewidth=linewidth_list[count], color=color_list[count])
        count += 1
    lg = ax.legend(fontsize=13)
    plt.savefig(fig_path)
    plt.show()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    download_csv_data_from_dev()
    for exp in range(1, 6):
        if exp == 1:
            kw_list = ["tape_aug", "tape_no_aug", "no_tape_aug", "no_tape_no_aug"]
            label_list = ["Full", "No augmentation", "No TAPE", "No augmentation, and No TAPE"]
        elif exp == 2:
            kw_list = ["tape_aug", "tape_aug_clip", "tape_aug_no_clip_augmultiply", "lnc_tape_aug"]
            label_list = ["Full", "Aug Clip", "Multiply", "lncRNA only"]
        elif exp == 3:
            kw_list = ["tape_aug", "tape_aug_only_rna", "tape_aug_only_pro", "lnc_unknown_tape_aug"]
            label_list = ["Full", "RNA model", "Protein model", "Unknown Protein(lncRNA)"]
        elif exp == 4:
            kw_list = ["tape_aug", "tape_aug_no_2dsoft", "tape_no_aug"]
            label_list = ["Full", "No 2D-softmax", "No augmentation"]
        elif exp == 5:
            kw_list = ["unknown0_tape_aug", "unknown1_tape_aug", "unknown2_tape_aug", "unknown3_tape_aug", "unknown4_tape_aug"]
            label_list = ["0", "1", "2", "3", "4"]

        for target in TARGETS:
                plot_from_csv(target, exp)
