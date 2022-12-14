# -------------------------------------------------------------------
# this code prepares rna files for fine tuning
# goal : raw sequences
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# input = "/Users/mac/Documents/transformer_tape_dnabert/data/GRCh38_latest_rna.fna"
input = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/benchmarks/sequence/"  # RPI369_rna_seq.fa"
output = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/benchmarks/sequence/all_bench_for_pretrain.csv"
# output = "/Users/mac/Documents/transformer_tape_dnabert/data/RNAseq_for_pretrain2.csv"
MAXLEN = 3999
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    with open(output, "w") as fo:
        for dirs in os.listdir(input):
            if "_rna_seq" in dirs:
                with open(f"{input}{dirs}") as f:
                    seq_str = ""
                    seq_list = []
                    for lines in f.readlines():
                        if lines[0] == ">":
                            seq_str = ""
                        else:
                            seq_str += lines.strip()
                            seq_list.append(seq_str)
                for item in seq_list:
                    for idx in range(len(item)):
                        if idx == len(item) - 1:
                            fo.writelines(f"{item[idx]}\n")
                        else:
                            fo.writelines(f"{item[idx]} ")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()