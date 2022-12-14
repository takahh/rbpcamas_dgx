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
from subprocess import call
import os
from multiprocessing import Pool

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
gencode = "/Users/mac/Documents/transformer_tape_dnabert/data/GENCODE_lncRNA_fasta/gencode.v40.lncRNA_transcripts.fa"
rbpsuite = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/RBPsuite_data/"  # AARS.negative.fa
outpath = "/Users/mac/Documents/transformer_tape_dnabert/data/lncRNA_in_RBPsuite/"
query_fastabase = "/Users/mac/Documents/query.fa"
logbase = "/Users/mac/Documents/blastn.log"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def select(cpunum):
    seq_count = 0
    log = f"{logbase}{cpunum}"
    query_fasta = f"{query_fastabase}{cpunum}"

    for idx, files in enumerate(os.listdir(rbpsuite)):
        if idx % 12 != cpunum:
            continue
        lnclist = []
        if ".DS" in files:
            continue
        with open(f"{rbpsuite}{files}") as f:
            for lines in f.readlines():
                if ">" in lines:
                    header = lines
                if ">" not in lines:
                    seq_count += 1
                    # make a query fasta file including one sequence
                    with open(f"{query_fasta}", "w") as fo:
                        fo.writelines(header + lines)
                    # run blastn
                    command = f"/usr/local/ncbi/blast/bin/blastn -subject {gencode} -query {query_fasta} > {log}"
                    call(command, shell=True)
                    # read blast stdout to check if the sequeuce is lncRNA or not
                    this_is_lncRNA = 1
                    with open(log) as flog:
                        for loglines in flog.readlines():
                            if "** No hits found *****" in loglines:
                                this_is_lncRNA = 0
                                break
                    if this_is_lncRNA == 1:
                        with open(log) as flog:
                            for loglines in flog.readlines():
                                if " Identities = " in loglines:  #  Identities = 60/64 (94%), Gaps = 0/64 (0%)
                                    denomi_count = loglines.split("/")[1].split()[0]
                                    identity_rate = loglines.split("%")[0].split("(")[-1]
                                    if int(denomi_count) > 90 and float(identity_rate) > 0.95:
                                        # print(f"denomi `{denomi_count}, identity {identity_rate}")
                                        lnclist.append(lines)
                                        print(len(lnclist))
                                    break
                if len(lnclist) > 90:
                    break

        with open(f"{outpath}{files}", "w") as fo:
            for item in lnclist:
                fo.writelines(item)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    p = Pool(12)
    p.map(select, range(12))
