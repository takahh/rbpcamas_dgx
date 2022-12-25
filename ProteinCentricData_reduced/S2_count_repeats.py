# -------------------------------------------------------------------
# this code counts for a specific protein shared by other proteins
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import pandas as pd
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------

# output data : chr19-944334-944429,117,"['WDR43', 'DDX6', ... 'EWSR1']"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def runs2():
    basepath = "/Users/mac/Documents/RBP_CAMAS/data/newdata/overlaps/"
    opath = "/Users/mac/Documents/RBP_CAMAS/data/newdata/counts/"
    path = f"{basepath}chr1/"  # binding sites of about 101 nt
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
                    for lines in f.readlines():
                        ele = lines.split()
                        # if no overlap, ignore
                        if len(ele[13]) < 3:
                            continue
                        # 0/10    1/11       2/12       3/13           4    5    6          7        8   9   20
                        # chr22  19480012  19480110  AATF_K562_rep01   200  +   1.797486   3.52936  -1  -1 \
                        # chr22  19480080  19480119  DDX21_K562_rep02  200  +   0.546817   0.28248　-1  -1   30
                        chromo_id = ele[0].split("_")[0]
                        # skip if the overlap partner (f=0.7) is longer than 80 nt
                        # if int(ele[20]) > 80:
                        #     continue
                        # if int(ele[2]) - int(ele[1]) + 1 > 101:
                        #     print(lines)
                        range_str = f"{chromo_id}-{ele[1]}-{ele[2]}"
                        protein_name = ele[13].split("_")[0]
                        # if "_2_" in protein_name:
                        #     print(f"range {range_str}, protein {protein_name}")
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
    runs2()