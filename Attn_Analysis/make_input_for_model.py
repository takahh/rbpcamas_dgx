# -------------------------------------------------------------------
# this code makes the very first data for preprocessing and prediction
# input : "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/data/pdbfiles"
# output: "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/data/for_model/seqences_and label"
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import numpy as np

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
input = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/data/pdbfiles/"
# output = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/data/for_model/seqences_and_label/"
pdbid_dict = ["6V5B_C_D", "6V5B_C_D", "6V5B_C_D", "6V5B_C_D", "6V5B_C_D"]
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    for mode in ["hb", "pi"]:
        datalist = []
        for dir in pdbid_dict:
            print(dir)
            if len(dir) != 8:
                continue
            seqs_for_model = f"{input}{dir}/{dir[:4]}_fasta_for_model_{mode}.csv"
            print(seqs_for_model)
            if not os.path.exists(seqs_for_model):
                print(f"{seqs_for_model} does not exist!!!")
                continue
            # 5AN9_fasta_for_model_hb.csv
            with open(seqs_for_model) as f:
                contents = f.read()
                pseq = contents.split(",")[0]
                rseq = contents.split(",")[1].strip()
                print(dir)

                datalist.append([
                    pdbid_dict.index(dir),
                    pseq,
                    rseq,
                    1
                ])
        output = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/training_data/attn_analysis_{mode}/"
        np.save(f"{output}0.npy", np.array(datalist))


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()