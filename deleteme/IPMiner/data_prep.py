# -------------------------------------------------------------------
# this code prepares dataset for IPMiner
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
import os
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
filecount = 1200
path = "/Users/mac/Documents/transformer_tape_dnabert/data/known_protein_lncRNA/"
opath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/IPMiner/ncRNA-protein/"
dirnames = [f"eclip{filecount}_rna.fa", f"eclip{filecount}_protein.fa", f"eclip{filecount}_NegativePairs.csv", f"eclip{filecount}_PositivePairs.csv", f"eclip{filecount}_all.txt"]
ofilenames = [f"{opath}{x}" for x in dirnames]
RNA_id_counter = [0] * 154
pro_seq_dict = {}

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    count = 0
    with open(ofilenames[0], "w") as f0, open(ofilenames[1], "w") as f1, open(ofilenames[2], "w") as f2,\
        open(ofilenames[3], "w") as f3, open(ofilenames[4], "w") as f4:
        f2.writelines(f"Protein ID\tRNA ID\n")
        f3.writelines(f"Protein ID\tRNA ID\n")
        # for files in os.listdir(path):
        for i in range(filecount):
            if count % 100 == 0:
                print(count)
            files = f"{i}.npy"
            arr = np.load(f"{path}{files}", allow_pickle=True)
            for item in arr:

                # retrieve variable values
                protein_id = int(item[0])
                protein_seq = item[1]
                RNA_seq = item[2]
                label = item[3]

                # update protein dictionary
                if protein_id not in pro_seq_dict.keys():
                    pro_seq_dict[str(protein_id)] = protein_seq
                # write RNA sequences
                RNA_id = f"{protein_id}_{RNA_id_counter[protein_id]}"
                f0.writelines(f">{RNA_id}\n{RNA_seq}\n")

                # write nega or posi bind pair
                if label == "1":  # positive - pro, RNA
                    f3.writelines(f"{protein_id}\t{RNA_id}\n")
                    f4.writelines(f"{protein_id}\t{RNA_id}\t1\n")
                else:
                    f2.writelines(f"{protein_id}\t{RNA_id}\n")
                    f4.writelines(f"{protein_id}\t{RNA_id}\t0\n")

                # increment RNA_id
                RNA_id_counter[protein_id] += 1
                count += 1
        # write protein_seq at the end
        for num in pro_seq_dict.keys():
            f1.writelines(f">{num}\n{pro_seq_dict[num]}\n")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()