# -------------------------------------------------------------------
# this code takes information from logs

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

# unknown_kw = "unk0red8_coefsum1.e"
unknown_kw = "unk0red8_no_no.e"

# dirpath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/single_node/"
dirpath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/single_node/"
dirpath2 = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/unknown_broad_share_red8/"
dirpath3 = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/"
figpath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/plot/optimization/"
exp_name_dict = {"tape_aug_only_pro": "Only_pro", "tape_aug_clip": "Clip_coeff", "tape_aug_only_rna": "Only_RNA",
                 "tape_aug_no_2dsfot": "1D_softmax", "tape_aug_augmultiply": "Aug_prod",
                 "no_tape_no_aug": "No_pre_no_aug", "tape_aug": "Full", "tape_no_aug": "No_aug",
                 "only_lncRNA": "lnc_RNA", "no_tape_aug": "No_pre"}

unknown_list = ["0", "1", "2", "3", "4"]

exp_name_dict_rev = {"Only_pro": "tape_aug_only_pro", "Clip_coeff": "tape_aug_clip", "Only_RNA": "tape_aug_only_rna",
                     "1D_softmax": "tape_aug_no_2dsfot", "Aug_prod": "tape_aug_augmultiply",
                     "No_pre_no_aug": "no_tape_no_aug", "Full": "tape_aug", "No_aug": "tape_no_aug",
                     "lnc_RNA": "only_lncRNA", "No_pre": "no_tape_aug"}

exp_name_dict_rev2 = {"No_pre_no_aug": "shared_4_no_pre_no_aug", "No_pre_aug": "shared_4_no_pre_aug"}

grouplist = [["Full", "No_pre", "No_aug", "No_pre_no_aug"], ["Full", "1D_softmax", "No_aug"],
             ["Full", "Only_RNA", "Only_pro"], ["Full", "Clip_coeff", "lnc_RNA", "Aug_prod"]]
grouplist2 = [["No_pre_aug", "No_pre_no_aug"]]


linestyle_list = ["-", ":", "-", "-", ":"]
linewidth_list = [3.5, 1.0, 1, 2, 2]
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------
plt.rcParams["font.size"] = 13


def avg2actnum(value, steps, sofar_sum):
    return value * steps - sofar_sum


def get_epoch_values(filenum, linelist, mode=None):  # get data "from 1 efile"
    if mode == "epoch":
        # Epoch 24, Loss: 0.6633608341217041, Train AUC: 10.311481475830078, Test Loss: 0.7021768093109131, Test AUC: 8.37075138092041
        train_loss_list = [float(x.split(" ")[3][:-1]) for x in linelist]
        test_loss_list = [float(x.split(" ")[9][:-1]) for x in linelist]
        return train_loss_list, test_loss_list
    else:
        subauclist, sublosslist = [], []
        coeff_0_att_mean, coeff_1_att_mean, coeff_0_aug_mean, coeff_1_aug_mean = [], [], [], []
        if mode == "coeff":
            filenum *= 2
        maxepoch = len(linelist)//filenum + 1
        epochcount = 0
        pred_arr = None  # for last epoch
        label_arr = None  # for last epoch
        for i in range(maxepoch - 1):  # epoch number is "i"
            start = filenum * i
            end = start + filenum
            totalstep = 0

            pred_arr = None  # for last epoch
            label_arr = None  # for last epoch
            if mode == "lossonly":
                tfloss_list = []
                for lines in linelist[start: end]:
                    totalstep += 1
                    tfloss_list.append(float(lines.split(",")[0].split(" ")[1]))
                sublosslist.extend(tfloss_list)

            elif mode == "coeff":
                inturn = 0
                coeff_list_0, coeff_list_1 = [], []
                for lines in linelist[start: end]:
                    totalstep += 1
                    # coeff att [1.0140448], coeff aug [1.0595027]
                    att = lines.split(" ")[2].replace("[", "").replace("]", "")
                    aug = lines.split(" ")[5].replace("[", "").replace("]", "")
                    if inturn == 0:
                        coeff_list_0.append([att, aug])
                        inturn = 1
                    else:
                        coeff_list_1.append([att, aug])
                        inturn = 0
                    if totalstep % filenum == 0:
                        coeff_0_att_list = [float(x[0].replace(",", "")) for x in coeff_list_0]
                        coeff_0_aug_list = [float(x[1].replace(",", "")) for x in coeff_list_0]
                        coeff_1_att_list = [float(x[0].replace(",", "")) for x in coeff_list_1]
                        coeff_1_aug_list = [float(x[1].replace(",", "")) for x in coeff_list_1]
                        coeff_0_att_mean.append(sum(coeff_0_att_list)/len(coeff_0_att_list))
                        coeff_0_aug_mean.append(sum(coeff_0_aug_list)/len(coeff_0_aug_list))
                        coeff_1_att_mean.append(sum(coeff_1_att_list)/len(coeff_1_att_list))
                        coeff_1_aug_mean.append(sum(coeff_1_aug_list)/len(coeff_1_aug_list))
                        coeff_list_0, coeff_list_1 = [], []

            else:
                for lines in linelist[start: end]:  # within 1 epoch
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
                        subauclist.append(metrics.roc_auc_score(label_arr[:, 1], pred_arr[:, 1]))
                        sublosslist.append(metrics.log_loss(label_arr, pred_arr))
                        epochcount += 1
                        totalstep = 0
        if mode == "lossonly":
            return sublosslist
        if mode == "coeff":
            return coeff_0_att_mean, coeff_1_att_mean, coeff_0_aug_mean, coeff_1_aug_mean
        else:
            return subauclist, sublosslist, pred_arr, label_arr


def get_plot_data(mode, logpath, fold=None, redlevel=None):  # {logpath}/tape_aug.e4323444, work for multiple files
    filecount, file_num_per_epoch = 0, 0
    coeff_0_att_list, coeff_0_aug_list, coeff_1_att_list, coeff_1_aug_list = [], [], [], []
    epoch_train_loss_list, epoch_test_loss_list = [], []
    keyw, keyw2, keyw3, keyw4 = None, None, None, None
    subauclist_all, sublosslist_all, pred_arr, label_arr = [], [], None, None
    tflosslist_all = []
    if mode == "Test":
        keyw = "test ROC"
        keyw3 = "testloss "
        if fold is None:
            file_num_per_epoch = 200
        else:
            file_num_per_epoch = 100
    elif mode == "Train":
        keyw = "train ROC"
        keyw2 = "coeff"
        keyw3 = "trainloss"
        keyw4 = "Epoch"
        if fold is None:
            file_num_per_epoch = 1000
        else:
            file_num_per_epoch = 300

    # --------------------------------------------------------
    # obtain information of e-files and sort them by date
    # --------------------------------------------------------
    if fold is not None:
        file_key = unknown_kw
    else:
        file_key = ".e"
    efilelist = [x for x in os.listdir(logpath) if file_key in x]
    datelist = [int(x.split(file_key)[1]) for x in os.listdir(logpath) if file_key in x]
    datelist.sort()

    # --------------------------------------------------------
    # get the last epoch labels & predictions, and all epoch auc and loss values
    # --------------------------------------------------------
    for datestr in datelist:
        for files in efilelist:
            if str(datestr) in files:
                linelist = []
                linelist2 = []
                with open(f"{logpath}{files}") as f:
                    linelist = [x.strip() for x in f.readlines() if keyw in x]
                with open(f"{logpath}{files}") as f:
                    losslist = [x.strip() for x in f.readlines() if keyw3 in x]
                print(files)
                print(linelist)
                if mode == "Train":
                    with open(f"{logpath}{files}") as f:
                        epoch_data_list = [x.strip() for x in f.readlines() if keyw4 in x]
                    # for lines in f.readlines():
                    #     if keyw in lines:
                    #         linelist.append(lines.strip())
                    #     if mode == "Train":
                    #         if keyw2 in lines:
                    #             linelist2.append(lines.strip())
                # collect date per file (about 30 epochs), list for 30 epochs, arr for 1 epoch
                subauclist, sublosslist, pred_arr, label_arr = get_epoch_values(file_num_per_epoch, linelist)
                tflosslist = get_epoch_values(file_num_per_epoch, losslist, "lossonly")

                if mode == "Train":
                    epoch_train_loss_list, epoch_test_loss_list = get_epoch_values(file_num_per_epoch, epoch_data_list, "epoch")
                if mode == "Train":
                    with open(f"{logpath}{files}") as f:
                        linelist2 = [x.strip() for x in f.readlines() if keyw2 in x]
                    head0_att_mean, head0_aug_mean, head1_att_mean, head1_aug_mean = get_epoch_values(file_num_per_epoch, linelist2, "coeff")
                    coeff_0_att_list.extend(head0_att_mean)
                    coeff_0_aug_list.extend(head0_aug_mean)
                    coeff_1_att_list.extend(head1_att_mean)
                    coeff_1_aug_list.extend(head1_aug_mean)
                subauclist_all += subauclist
                sublosslist_all += sublosslist
                tflosslist_all += tflosslist
                filecount += 1
    if mode == "Train":
        return [sublosslist_all, subauclist_all, label_arr, pred_arr, [coeff_0_att_list, coeff_0_aug_list, coeff_1_att_list, coeff_1_aug_list]
                , tflosslist_all, [epoch_train_loss_list, epoch_test_loss_list]]
    else:
        return [sublosslist_all, subauclist_all, label_arr, pred_arr, tflosslist_all]


def group_plot(plotdata, phasetype, expnames, gr_idx):
    # plotdata = [{"Train": [losslist, auclist, labels, preds, coeff0, coeff1], "Test": [losslist, auclist, labels, preds]},
    #             {Train": [losslist, auclist, labels, preds, coeff0, coeff1], "Test": [losslist, auclist, labels, preds]}, ...]

    # --------------------------------------------------------
    # COEFF plot
    # --------------------------------------------------------
    if phasetype == "Train":
        plt.figure()   # coeff_0_att_mean, coeff_1_att_mean, coeff_0_aug_mean, coeff_1_aug_mean
        coeff_labels = ["Head 0, Attention Coeff.",  "Head 1, Attention Coeff.", "Head 0, Augmentation Coeff.", "Head 1, Augmentation Coeff."]
        for idx, sublist in enumerate(plotdata):  # plotdata : data for 1 task
            exp = expnames[idx]
            if "no_aug" in exp or "No_aug" in exp:
                continue
            for i in range(4):
                plt.plot(range(len(sublist[phasetype][4][i])), sublist[phasetype][4][i], label=coeff_labels[i], linestyle=linestyle_list[i],
                        linewidth=linewidth_list[i])
            fig_name = f"group_{gr_idx}_{phasetype}_{exp}_coeff.png"
            plt.title(f"{exp} Coefficient")
            plt.ylabel("Value", fontsize=15)
            plt.xlabel("Epoch", fontsize=15)
            plt.legend()
            plt.savefig(f"{figpath}{fig_name}")
            plt.show()


    # --------------------------------------------------------
    # LOSS plot
    # --------------------------------------------------------
    fig, ax = plt.subplots()
    for idx, sublist in enumerate(plotdata):
        ax.plot(range(len(sublist[phasetype][0])), sublist[phasetype][0], label=expnames[idx], linestyle=linestyle_list[idx],
                linewidth=linewidth_list[idx])
    ax.grid(visible=True, axis="y")
    fig_name = f"group_{gr_idx}_{phasetype}_loss.png"
    # plt.title(f"{phasetype} Loss")
    plt.ylabel(f"{phasetype} Loss", fontsize=15)
    plt.xlabel("Epoch", fontsize=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figpath}{fig_name}")
    plt.show()

    # --------------------------------------------------------
    # AUC plot
    # --------------------------------------------------------
    fig, ax = plt.subplots()
    for idx, sublist in enumerate(plotdata):
        plt.plot(range(len(sublist[phasetype][1])), sublist[phasetype][1], label=expnames[idx], linestyle=linestyle_list[idx],
                linewidth=linewidth_list[idx])
    ax.grid(visible=True, axis="y")
    # plt.title(f"{phasetype} AUROC")
    plt.ylabel("AUROC", fontsize=15)
    plt.xlabel("Epoch", fontsize=15)
    fig_name = f"group_{gr_idx}_{phasetype}_AUROC.png"
    plt.ylim(0.5, 1)
    ax.legend()
    plt.savefig(f"{figpath}{fig_name}")
    plt.show()

    # --------------------------------------------------------
    # ROC plot
    # --------------------------------------------------------
    fig, ax = plt.subplots()
    for idx, sublist in enumerate(plotdata):
        label_arr = sublist[phasetype][2]
        pred_arr = sublist[phasetype][3]
        rocd = roc_curve(label_arr[:, 1], pred_arr[:, 1])
        rocscore = roc_auc_score(label_arr[:, 1], pred_arr[:, 1])
        ax.plot(rocd[0], rocd[1], label=f"{expnames[idx]} ({round(rocscore, 4)})", linestyle=linestyle_list[idx],
                linewidth=linewidth_list[idx])
    fig_name = f"group_{gr_idx}_{phasetype}_AUROC.png"
    # plt.title(f"{phasetype} ROC")
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.legend()
    plt.savefig(f"{figpath}{fig_name}")
    plt.show()
    #
    # # --------------------------------------------------------
    # # TF LOSS plot
    # # --------------------------------------------------------
    # if phasetype == "Train":
    #     list_idx = 5
    # else:
    #     list_idx = 4
    # fig, ax = plt.subplots()
    # for idx, sublist in enumerate(plotdata):
    #     ax.plot(range(len(sublist[phasetype][list_idx])), sublist[phasetype][list_idx], label=expnames[idx], linestyle=linestyle_list[idx],
    #             linewidth=linewidth_list[idx])
    # ax.grid(visible=True, axis="y")
    # fig_name = f"group_{gr_idx}_{phasetype}_tfloss.png"
    # # plt.title(f"{phasetype} Loss")
    # plt.ylabel("TF Loss", fontsize=15)
    # plt.xlabel("STEP", fontsize=15)
    # ax.legend()
    # plt.savefig(f"{figpath}{fig_name}")
    # plt.show()

    #
    # # --------------------------------------------------------
    # # EPOCH LOSS plot
    # # --------------------------------------------------------
    # # plot "Epoch 24, Loss: 0.663444, Train AUC: 10.311481475, Test Loss: 0.7021109131, Test AUC: 8.37075041
    # if phasetype == "Train":
    #     expnames = ["Train epoch loss", "Test epoch loss"]
    #     fig, ax = plt.subplots()
    #     for j in range(2):
    #         ax.plot(range(len(sublist[phasetype][6][j])), sublist[phasetype][6][j], label=expnames[j], linestyle=linestyle_list[j],
    #                 linewidth=linewidth_list[j])
    #     ax.grid(visible=True, axis="y")
    #     fig_name = f"group_{gr_idx}_{phasetype}_epoch_loss.png"
    #     # plt.title(f"{phasetype} Loss")
    #     plt.ylabel("Epoch Train Loss", fontsize=15)
    #     plt.xlabel("EPOCH", fontsize=15)
    #     ax.legend()
    #     plt.savefig(f"{figpath}{fig_name}")
    #     plt.show()


def reduced():
    # -----------------------------
    # Ablation
    # -----------------------------
    for gidx, explist in enumerate(grouplist2):
        datalist = []

        # obtain data for plot
        for expname in explist:  # expname e.g. Full
            dirname = exp_name_dict_rev2[expname]
            plot_data_dict = {}
            for phase in ["Train", "Test"]:
                plot_data_dict[phase] = get_plot_data(phase, f"{dirpath3}{dirname}/")
            datalist.append(plot_data_dict)

        # plot for each phase(Train or Test)
        for phase in ["Train", "Test"]:
        # for phase in ["Train"]:
            group_plot(datalist, phase, explist, gidx)



def ablation():
    # -----------------------------
    # Ablation
    # -----------------------------
    for gidx, explist in enumerate(grouplist):
        datalist = []

        # obtain data for plot
        for expname in explist:  # expname e.g. Full
            dirname = exp_name_dict_rev[expname]
            plot_data_dict = {}
            for phase in ["Train", "Test"]:
                plot_data_dict[phase] = get_plot_data(phase, f"{dirpath}{dirname}/")
            datalist.append(plot_data_dict)

        # plot for each phase(Train or Test)
        for phase in ["Train", "Test"]:
        # for phase in ["Train"]:
            group_plot(datalist, phase, explist, gidx)


def unknown():
    # -----------------------------
    # Unknown Proteins
    # -----------------------------
    # obtain data for plot
    datalist = []
    reduce_level = 8
    for foldid in range(1):  # perrange experiment such as Full
        plot_data_dict = {}
        for phase in ["Train", "Test"]:
            plot_data_dict[phase] = get_plot_data(phase, f"{dirpath2}", foldid, reduce_level)
        datalist.append(plot_data_dict)
    # plot for each phase(Train or Test)
    for phase in ["Train", "Test"]:
        group_plot(datalist, phase, unknown_list, 4)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Normal
    ablation()
    # unknown()
    # reduced()
