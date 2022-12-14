# -------------------------------------------------------------------
# this code makes a data file of 1mer
input = "/Users/mac/Documents/transformer_tape_dnabert/DNABERT/examples/sample_data/pre/6_3k.txt"
output1 = "/Users/mac/Documents/transformer_tape_dnabert/DNABERT/examples/sample_data/pre/rawseq.txt"
output2 = "/Users/mac/Documents/transformer_tape_dnabert/DNABERT/examples/sample_data/pre/1_3k.txt"
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
from motif_utils import seq2kmer, kmer2seq

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    with open(output2, "w") as fo:
        with open(input) as f:
            for line in f.readlines():
                fo.writelines(f"{seq2kmer(kmer2seq(line.strip()), 1)}\n")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()