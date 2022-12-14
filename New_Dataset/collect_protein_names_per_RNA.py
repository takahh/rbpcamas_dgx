# -------------------------------------------------------------------
# this code collects protein names per RNA sequence after running
# "findsame_RNA_seqs.py"
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
import os
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/common_search_60000/"
opath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/bind_data_per_RNA/"
# file = f"{path}AARS.negative.fa/AARS.negative.fa_AARS.positive.fa"
label_dict = {"positive": 1, "negative": 0}
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def already_in_file(pronameline, ofile_path):
    with open(ofile_path) as f:
        if pronameline in f.readlines():
            return True
        else:
            return False


def main():
    countprotein = 0
    for dirs in os.listdir(path):
        print(countprotein)
        countprotein += 1
        for files in os.listdir(f"{path}{dirs}"):
            # files : AARS.negative.fa_AARS.positive.fa
            protein1 = files.split(".")[0]
            protein1_label = label_dict[files.split(".")[1]]
            protein2 = files.split(".")[2].replace("fa_", "")
            protein2_label = label_dict[files.split(".")[3]]
            line_to_add_1 = f"{protein1},{protein1_label}\n"
            line_to_add_2 = f"{protein2},{protein2_label}\n"
            with open(f"{path}{dirs}/{files}") as f:
                seqlist = [x.strip() for x in f.readlines()]
                for seq in seqlist:
                    foldername = seq[:5]
                    if not os.path.exists(f"{opath}{foldername}"):
                        os.mkdir(f"{opath}{foldername}")
                    try:
                        if os.path.exists(f"{opath}{foldername}/{seq}"):
                            with open(f"{opath}{foldername}/{seq}", "a") as fo:
                                if not already_in_file(line_to_add_1, f"{opath}{foldername}/{seq}"):
                                    fo.writelines(line_to_add_1)
                                if not already_in_file(line_to_add_2, f"{opath}{foldername}/{seq}"):
                                    fo.writelines(line_to_add_2)
                        else:
                            with open(f"{opath}{foldername}/{seq}", "w") as fo:
                                if not already_in_file(line_to_add_1, f"{opath}{foldername}/{seq}"):
                                    fo.writelines(line_to_add_1)
                                if not already_in_file(line_to_add_2, f"{opath}{foldername}/{seq}"):
                                    fo.writelines(line_to_add_2)
                    except OSError:
                        print(f"{foldername}/{files}")
        #         break
        # break


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()