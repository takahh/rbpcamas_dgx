# -------------------------------------------------------------------
# this code counts shared signals per chromoN and Protein
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import pandas as pd
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
basepath = "/Users/mac/Documents/RBP_CAMAS/data/newdata/overlaps/"
opath = "/Users/mac/Documents/RBP_CAMAS/data/newdata/counts/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    path = f"{basepath}chr1/"
    for proteins in os.listdir(path):
        if ".tsv" in proteins or ".txt" in proteins or ".DS" in proteins or ".bed" in proteins:
            continue
        else:
            range_dict = {}
            count_dict = {}
            path1 = f"{path}{proteins}/"
            for files in os.listdir(path1):  # ENCFF005ZCI.bed"
                if ".DS" in files:
                    continue
                with open(f"{path1}{files}") as f:
                    print(f"checking {files}...")
                    for lines in f.readlines():
                        ele = lines.split()
                        if len(ele[13]) == 1:
                            continue
                        # chr22   19480012        19480110        AATF_K562_rep01 200     +       1.79748642736451
                        # 3.52936903631509        -1      -1      chr22   19480080        19480119
                        # DDX21_K562_rep02        200     +       0.546817334292526       0.282489350110395
                        # -1      -1      30
                        chromo_id = ele[0].split("_")[0]
                        range_str = f"{chromo_id}-{ele[1]}-{ele[2]}"
                        protein_name = ele[13].split("_")[0]
                        if "_2_" in protein_name:
                            print(f"range {range_str}, protein {protein_name}")
                        if range_str in range_dict.keys():
                            if protein_name in range_dict[range_str]:
                                pass
                            else:
                                range_dict[range_str].append(protein_name)
                        else:
                            range_dict[range_str] = [protein_name]
                            # range_dict[range_str].append(protein_name)
                # print(range_dict)
                for key in range_dict.keys():
                    count_dict[key] = [len(range_dict[key]), range_dict[key]]
            # print(count_dict)
            df = pd.DataFrame.from_dict(count_dict, orient="index")
            df.to_csv(f"{opath}/{proteins}.csv")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()