# -------------------------------------------------------------------
# this code runs blastn locally to select lncRNA sequences
# from RBPsuite
# input
#   GENCODE lncRNA : /Users/mac/Documents/transformer_tape_dnabert/data/GENCODE_lncRNA_fasta/gencode.v40.lncRNA_transcripts.fa
#   RBPsuite : /Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/RBPsuite_data/AARS.negative.fa
# output:
#   /Users/mac/Documents/transformer_tape_dnabert/data/lncRNA_in_RBPsuite/
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import random
import subprocess
from subprocess import call
import os
from multiprocessing import Pool
import argparse

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/"
rbpsuite = f"{path}RBPsuite_data/"  # AARS.negative.fa

# rbpsuite = "/Users/mac/Documents/transformer_tape_dnabert/train_dir/"
# path = "/Users/mac/Documents/transformer_tape_dnabert/data/"

gencode = f"{path}GENCODE_lncRNA_fasta/gencode.v40.lncRNA_transcripts.fa"
outpath = f"{path}lncRNA_in_RBPsuite/"
query_fastabase = f"{path}query.fa"
logbase = f"{path}blastn.log"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def select(cpunum):
    seq_count = 0
    log = f"{logbase}{cpunum}"
    query_fasta = f"{query_fastabase}{cpunum}"

    for idx, files in enumerate(os.listdir(rbpsuite)):
        if idx % 56 != cpunum:
            continue
        lnclist = []
        if ".DS" in files:
            continue

        # shuffle entries in a file
        with open(f"{rbpsuite}{files}") as f:
            itemlist = f.read().split("\n>")
            random.shuffle(itemlist)

        # run blastn for each sequence
        for item in itemlist:
            element = item.split("\n")
            if element[0][0] != ">":
                header = f">{element[0]}\n"
            else:
                header = f"{element[0]}\n"
            seq_count += 1
            # make a query fasta file including one sequence
            with open(f"{query_fasta}", "w") as fo:
                fo.writelines(header + element[1] + "\n")

            # run blastn
            command = f"blastn -subject {gencode} -query {query_fasta}"
            response = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read().decode("ascii")

            # read blast stdout to check if the sequeuce is lncRNA or not
            if "** No hits found *****" in response:
                this_is_lncRNA = 0
                continue
            else:  #  Identities = 60/64 (94%), Gaps = 0/64 (0%)
                toread = response.split("Identities = ")[1].split(",")[0]  # 60/64 (94%)
                denomi_count = toread.split("/")[1].split()[0]
                identity_rate = toread.split("%")[0].split("(")[-1]
                if int(denomi_count) > 90 and float(identity_rate) > 0.95:
                    # print(f"denomi `{denomi_count}, identity {identity_rate}")
                    lnclist.append(element[1] + "\n")
                    # print(len(lnclist))
                    if len(lnclist) > 90:
                        break

        with open(f"{outpath}{files}", "w") as fo:
            for item in lnclist:
                fo.writelines(item)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r')
    p = Pool(28)
    args = parser.parse_args()
    start = int(args.r)
    p.map(select, range(start, start + 28))
