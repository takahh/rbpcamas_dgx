# -------------------------------------------------------------------
# this code makes two hetamaps for physical contact map and attention map
# input :
#   phys_map : /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/physical_interaction_map/hb_5AN9_J_N.npz
#   attn_map : /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/output_weight/attn_analysis_hb.npy

# output: /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/plots/
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os.path

import numpy as np
import seaborn
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 14
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)
# array_out_path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/physical_interaction_map/" previous script
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------

RMIN, RMAX = 50, 75
# RMIN, RMAX = 0, 101
# PMIN, PMAX = 0, 1981
PMIN, PMAX = 0, 25
# PDBID = "6FF4_A_6_from_head"
# PDBID = "6FF4_A_6_from_tail"
PDBID = "6V5B_C_D"
pdbid_dict = [PDBID] * 5
# pdbid_dict = ["5AN9_J_N", "6ZLW_F_2", "6ZLW_K_2", "6ZXG_L_2", "6ZXG_L_2"]
attnpath_self_dir = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/output_weight/{PDBID}/"
modelist = ["hb", "pi"]
PATH = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/"
path = f"{PATH}physical_interaction_map"
# sample pysicalpath = f"{path}/hb_5AN9_J_N.npz"
figpath = f"{PATH}plot/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_seq_loc_lists(pid, btype):
    print(btype)
    true_fasta = f"{PATH}data/pdbfiles/{pid}/{pid[:4]}_tru_fasta_hb.csv"
    with open(true_fasta) as f:
        r_loc_list = [x.split(",")[0] for x in f.readlines() if len(x.split(",")[1].strip()) == 1]
    with open(true_fasta) as f:
        p_loc_list = [x.split(",")[0] for x in f.readlines() if len(x.split(",")[1]) == 3]
        print(len(p_loc_list))
    return p_loc_list, r_loc_list


def get_protein_length(pairid):
    path = f"{PATH}data/pdbfiles/{pairid}/{pairid[:4]}_fasta_for_model_pi.csv"
    with open(path) as f:
        return len(f.read().split(",")[0])


def heatmap(arr_data, chdata, pathid, bondtype=None):
    df_zoomed = pd.DataFrame([])
    targetdir = f"{figpath}{chdata}"
    if not os.path.exists(targetdir):
        os.mkdir(targetdir)
    if bondtype == "hb":
        title = f"{chdata}, Hydrogen Bond, {pathid}\n"
    elif bondtype == "pi":
        title = f"{chdata}, Pi Interaction, {pathid}\n"
    else:
        title = f"{chdata}, {pathid}\n"
    plt.figure()
    f, ax = plt.subplots()
    # plt.title(title, fontsize=15)
    plt.gcf().subplots_adjust(bottom=0.15)
    protein_length = get_protein_length(chdata)
    print(f"{chdata}, {protein_length}")

    # clip dataframe if protein is included
    if bondtype:  # from PDB
        ploc_list, rloc_list = get_seq_loc_lists(chdata, bondtype)
        print(len(rloc_list))
        print(arr_data.shape)
        df = pd.DataFrame(arr_data[:protein_length, :], columns=rloc_list, index=ploc_list)
        # df = pd.DataFrame(arr_data[:protein_length, :])
    else:  # from model
        ploc_list, rloc_list = get_seq_loc_lists(chdata, "hb")

        print(f"ploc {len(ploc_list)}, rloc {rloc_list}")
        if "Protein Self" in title:
            df = pd.DataFrame(arr_data[:protein_length, :protein_length], columns=ploc_list, index=ploc_list)
        elif "RNA Self" in title:
            df = pd.DataFrame(arr_data[:, :], columns=rloc_list, index=rloc_list)
        else:  # "Cross Attention"
            df = pd.DataFrame(arr_data[:protein_length, :], columns=rloc_list, index=ploc_list)
    if "6FF4" in chdata and "Self" not in title:
        df_zoomed = pd.DataFrame(arr_data[PMIN:PMAX, RMIN:RMAX], columns=rloc_list[RMIN: RMAX], index=ploc_list[PMIN:PMAX])
    if "6V5B" in chdata and "Self" not in title:
        df_zoomed = pd.DataFrame(arr_data[PMIN:PMAX, RMIN:RMAX], columns=rloc_list[RMIN: RMAX], index=ploc_list[PMIN:PMAX])

    if bondtype:  # from PDB
        ax = seaborn.heatmap(df, vmin=arr_data.min(), vmax=1, cmap="Greys", cbar=True)
        # ax = seaborn.heatmap(df, vmin=arr_data.min(), vmax=arr_data.max(), cmap="Greys", cbar=True)
        ax.tick_params(axis='x', rotation=90, pad=10)
        ax.set_ylabel("Protein", fontsize=16)
        ax.set_xlabel("RNA", fontsize=16)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(labelbottom=False, labeltop=True)
        ax.tick_params(axis='both', which='both', length=0)
        plt.tight_layout()
        plt.savefig(f"{targetdir}/{bondtype}_{pathid.replace(',', '').replace(' ', '_')}.png")
    elif "Protein Self" in title or "RNA Self" in title:
        labelstr = title.split(", ")[3].split()[0]
        ax = seaborn.heatmap(df, vmin=arr_data.min(), vmax=arr_data.max(), cmap="Greys", cbar=True)
        ax.tick_params(axis='x', rotation=90, pad=10)
        ax.set_ylabel(labelstr, fontsize=16)
        ax.set_xlabel(labelstr, fontsize=16)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(labelbottom=False, labeltop=True)
        ax.tick_params(axis='both', which='both', length=0)
        plt.tight_layout()
        plt.savefig(f"{targetdir}/{pathid.replace(',', '').replace(' ', '_')}.png")
    else:  # CROSS
        def plot_cross_attn(dataframe, original_arr, title2, outpath, keywd=None):
            if keywd:
                ax = seaborn.heatmap(dataframe, vmin=original_arr.min(), vmax=original_arr.max(), cmap="Greys", cbar=False)
            else:
                ax = seaborn.heatmap(dataframe, vmin=original_arr.min(), vmax=original_arr.max(), cmap="Greys", cbar=True)

            ax.tick_params(axis='x', rotation=90, pad=10)
            ax.set_ylabel("Protein", fontsize=16)
            ax.set_xlabel("RNA", fontsize=16)
            ax.xaxis.set_label_position('top')
            ax.xaxis.set_ticks_position('top')
            ax.tick_params(labelbottom=False, labeltop=True)
            ax.tick_params(axis='both', which='both', length=0)
            plt.tight_layout()
            if keywd:
                plt.savefig(f"{outpath}/{title2.replace(',', '').replace(' ', '_')}_p_{PMIN}_{PMAX}_r_{RMIN}_{RMAX}_zoomed.png")
            else:
                plt.savefig(f"{outpath}/{title2.replace(',', '').replace(' ', '_')}.png")
        plot_cross_attn(df, arr_data, pathid, targetdir)
        if df_zoomed.shape[0] > 0:
            plot_cross_attn(df_zoomed, arr_data, pathid, targetdir, "zoomed")

    plt.show()


def plot_from_model(pidx, dataid):

    # --------------------------
    # 1. first attn weight (multiplied with coeff)
    # 2. stat pots added  (multiplied with coeff)
    # 3. final attn wtight
    # --------------------------
    for chpoint in ["0", "1"]:
        for type in ["no_tape_aug", "no_tape_no_aug"]:
            ch_type = f"_{chpoint}_{type}"
            basepath = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/output_weight/{PDBID}/"
            filelist = [[f"added_pot_cross_pro{ch_type}.npy", f"added_pot_cross_rna{ch_type}.npy"],
                        [f"final_cross_pro{ch_type}.npy", f"final_cross_rna{ch_type}.npy"],
                        [f"first_cross_pro{ch_type}.npy", f"first_cross_rna{ch_type}.npy"]]
            datatypelist = ["Added Potential", "After Addition", "Before Addition"]

            for flist, dtype in zip(filelist, datatypelist):

                arr_p = np.load(f"{basepath}{flist[0]}", allow_pickle=True)
                arr_r = np.load(f"{basepath}{flist[1]}", allow_pickle=True)
                for head, mode in enumerate(modelist):  # hb, pi
                    sub_arr_p = arr_p[pidx, head, :, :]  # here the indx can be the other
                    heatmap(np.transpose(sub_arr_p), dataid, f"{dtype} Head {head}, Protein Path, from Model {ch_type}")
                    sub_arr_r = arr_r[pidx, head, :, :]  # here the indx can be the other
                    heatmap(sub_arr_r, dataid, f"{dtype} Head {head}, RNA Path, from Model {ch_type}")


def self_plot_from_model(pidx, dataid):
    for layer_num in range(4):
        # attn_analysis_hb_pro_self_l_1.npy
        arr_p = np.load(f"{attnpath_self_dir}attn_analysis_hb_pro_self_l_{layer_num + 1}.npy", allow_pickle=True)
        arr_r = np.load(f"{attnpath_self_dir}attn_analysis_hb_rna_self_l_{layer_num + 1}.npy", allow_pickle=True)
        for head, mode in enumerate(modelist):  # hb, pi
            sub_arr_p = arr_p[pidx, head, :, :]  # here the indx can be the other
            heatmap(np.transpose(sub_arr_p), dataid, f"Head {head}, Layer {layer_num + 1}, Protein Self, from Model")
            sub_arr_r = arr_r[pidx, head, :, :]  # here the indx can be the other
            heatmap(sub_arr_r, dataid, f"Head {head}, Layer {layer_num + 1}, RNA Self, from Model")


def plot_actual_interactionmap(did):
    for mode in ["hb", "pi"]:
        physicalpath = f"{path}/{mode}_{did}.npz"
        arr = np.load(physicalpath)
        # ['interactions', 'rna_true_list', 'pro_true_list']
        heatmap(arr["interactions"], did, "from PDB", mode)


def plot_raw(did):
    for mode in ["pro", "RNA"]:
        plt.figure()
        arr_path = f"//Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/output_weight/{data_id}/attn_analysis_hb_{mode}out.npy"
        arr = np.load(arr_path)
        print(arr.shape)
        if mode == "RNA":
            df = pd.DataFrame(arr[0, 0, :, :])
        else:
            df = pd.DataFrame(arr[0, :, :])
        print(mode)
        print(df.shape)
        ax = seaborn.heatmap(df, cmap="PuOr_r", cbar=True)
        ax.tick_params(axis='x', rotation=90, pad=10)
        # ax.set_ylabel("Protein", fontsize=16)
        # ax.set_xlabel("RNA", fontsize=16)
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(labelbottom=False, labeltop=True)
        ax.tick_params(axis='both', which='both', length=0)
        plt.tight_layout()
        plt.savefig(f"{figpath}/{did}/{mode}_raw.png")
        print("here")


def plot_stat_pots(protein_index, da_id):
    for mode in ["hb", "pi"]:
        arr_path = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/attn_arrays_{mode}/attn_ana/0.npz"
        arr = np.load(arr_path)["pot"]
        heatmap(arr[protein_index, :, :], da_id, " Statistical Potentials", mode)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':

    # Attention Map from Model Output
    for idx, data_id in enumerate(pdbid_dict):
        # plot CROSS attention matrix from predictions
        plot_from_model(idx, data_id)

        # # Interaction Map from Crystals etc.
        # plot_actual_interactionmap(data_id)

        # # statistical potential plot
        # plot_stat_pots(idx, data_id)
        #
        # # plot SELF attention matrix from predictions
        # self_plot_from_model(idx, data_id)
        #
        # # # plot raw outputs
        # plot_raw(data_id)

        break

