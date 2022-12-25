# -------------------------------------------------------------------
# this code collects negatives and run MMseqs2
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import random
# -------------------------------------------------------------------
# constant
# ----------------------------------------------------------------
path = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq25_32.csv"
path2 = "/Users/mac/Documents/RBP_CAMAS/data/newdata/counts/"
opath = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq25_32_with_negatives.csv"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_proteins():
    protein_list = []
    for dirs in os.listdir(path2):
        protein_list.append(dirs.replace(".csv", ""))
    protein_list = list(set(protein_list))
    return protein_list


def runs4(least_posi_count_per_rna=10):
    all_protein_list = get_proteins()
    with open(path) as f, open(opath, "w") as fo:
        for lines in f.readlines():
            ele = lines.split("[")
            # -----------------
            # get positive list
            # -----------------
            firststring = (ele[0][:-2].split(","))
            protein_list = [x.replace("]", "").replace("'", "") if "]" in x else x for x in ele[1].split(",")]
            protein_list = [x.replace("'", "").replace(" ", "").replace('"\n', '') for x in protein_list]
            # -----------------
            # get negative list
            # -----------------
            gap_list = list(set(all_protein_list).difference(set(protein_list)))
            posicount = len(protein_list)
            negacount = len(gap_list)
            # if posicount + negacount != 150:
            #     print(lines)
            # both should be greater than 20
            print(posicount)
            if posicount < least_posi_count_per_rna or negacount < least_posi_count_per_rna:
                continue
            if ".DS_Store" in gap_list:
                gap_list.remove(".DS_Store")
            print(f"posi {(posicount)}, nega {(negacount)}")
            random.shuffle(gap_list)
            random.shuffle(protein_list)
            # if posicount > negacount:
            gap_list = gap_list[:least_posi_count_per_rna]
            protein_list = protein_list[:least_posi_count_per_rna]
            # else:
            #     gap_list = gap_list[:posicount]
            #     protein_list = protein_list[:posicount]
            print(len(gap_list))
            print(len(protein_list))
            # print(f"{','.join(firststring)},{protein_list},{gap_list}\n")
            fo.writelines(f"{','.join(firststring)},{protein_list},{gap_list}\n")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    runs4()

