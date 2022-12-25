# -------------------------------------------------------------------
# this code collects negatives and run MMseqs2
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq20_25.csv"
path2 = "/Users/mac/Documents/RBP_CAMAS/data/newdata/counts/"
opath = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq20_25_with_negatives.csv"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_proteins():
    protein_list = []
    for dirs in os.listdir(path2):
        protein_list.append(dirs.replace(".csv", ""))
    protein_list = list(set(protein_list))
    return protein_list


def main():
    all_protein_list = get_proteins()
    with open(path) as f, open(opath, "w") as fo:
        for lines in f.readlines():
            ele = lines.split("[")
            protein_list = [x.replace("]", "").replace("'", "") if "]" in x else x for x in ele[1].split(",")]
            protein_list = [x.replace("'", "").replace(" ", "").replace('"\n', '') for x in protein_list]
            gap_list = set(all_protein_list).difference(set(protein_list))
            print(f"{len(all_protein_list)} - {len(protein_list)} = {len(gap_list)}")
            lines = lines.replace("\n", "")
            fo.writelines(f"{lines},{gap_list}\n")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()