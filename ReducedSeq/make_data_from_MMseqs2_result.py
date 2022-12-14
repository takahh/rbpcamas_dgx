# -------------------------------------------------------------------
# this code summarizes the result of MMseq2 for common RNA sequences
# and make input data for models
# run this script after mmseqs2forCommon.py
path = f"/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/cluster_reps/"
input = f"{path}5000/result/"  # LIN28B/positive_ZC3H11Apositivesummary.m8_small"
opath = f"{path}5000/newdata_distributed/"
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import numpy as np
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_list():  # per query protein
    # protein_name = "LIN28B"

    # ------------------------
    # collect sequences per query protein
    # ------------------------
    dict_to_write = {}
    for protein_name in os.listdir(input):
        if not os.path.isdir(f"{input}{protein_name}"):
            continue
        aggre_dict = {}
        # ------------------------
        # choose best 20 sequences
        # ------------------------
        for dirs in os.listdir(f"{input}{protein_name}"):
            if "small" not in dirs:
                continue
            with open(f"{input}{protein_name}/{dirs}") as f:
                for lines in f.readlines():
                    # sort first, score is ratio * length
                    ele = lines.split()
                    try:
                        score = float(ele[4]) * float(ele[5])
                    except IndexError:
                        print("IndexError !!!! ################3")
                        print(f"{input}{protein_name}/{dirs}")
                        print(lines)
                    #  key: own sequence id, value: target(label, protein name, sequence id )
                    if score > 100:
                        if ele[0] in aggre_dict.keys():
                            aggre_dict[ele[0]] += 1
                        else:
                            aggre_dict[ele[0]] = 1
        aggre_dict = {k: v for k, v in sorted(aggre_dict.items(), key=lambda item: item[1], reverse=True)}
        best30 = list(aggre_dict.keys())[:30]
        # ------------------------
        # distribute sequences
        # ------------------------
        for dirs in os.listdir(f"{input}{protein_name}"):
            if "small" not in dirs:
                continue
            # positive_AARSnegativesummary.m8_small
            qlabel = dirs.split("_")[0]  # positive
            if qlabel == "nagative":
                qlabel = "negative"
            target_name = dirs.split("_")[1].split("summary")[0]  # AARSnegative
            query_name = f"{protein_name}{qlabel}"
            with open(f"{input}{protein_name}/{dirs}") as f:
                for lines in f.readlines():
                    # 0      1        2       3       4  5       6
                    # 310625 ACA...CC 1907228 CG..UGG 96 0.916   0
                    # sort first, score is ratio * length
                    ele = lines.split()
                    try:
                        score = float(ele[4]) * float(ele[5])
                    except IndexError:
                        print("IndexError !!!! ################3")
                        print(f"{input}{protein_name}/{dirs}")
                        print(lines)
                    #  key: own sequence id, value: target(label, protein name, sequence id )
                    if score > 100 and ele[0] in best30:
                        #  key: own sequence id, value: target(label, protein name, sequence id )
                        # write query sequence
                        if query_name in dict_to_write.keys():
                            dict_to_write[query_name].append(ele[1])
                        else:
                            dict_to_write[query_name] = [ele[1]]
                        # write target sequence
                        if target_name in dict_to_write.keys():
                            dict_to_write[target_name].append(ele[3])
                        else:
                            dict_to_write[target_name] = [ele[3]]
    for key, value in dict_to_write.items():
        np.save(f"{opath}/{key}", value)


def main():
    get_list()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()
