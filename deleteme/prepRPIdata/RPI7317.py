# -------------------------------------------------------------------
# this code tranform name of RNA or protein to sequences
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/LGFC-CNN/LGFC-CNN/Datasets/Train_dataset/NPinter_human/"
pairlist = f"{path}RPI7317.txt"
proseqlist = f"{path}protein_human_fasta.fasta"
rnaseqlist = f"{path}RNA_human_fasta.fasta"
output = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/myRPI/RPI7317/allseqpairs.csv"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------

def getseqdict(filepath):
    seqdict = {}
    with open(filepath) as f:
        for lines in f.readlines():
            if ">" in lines:
                genename = lines[1:].strip()
            else:
                seqdict[genename] = lines.strip()
    return seqdict


def main():
    rnaseqdict = getseqdict(rnaseqlist)
    proseqdict = getseqdict(proseqlist)
    with open(pairlist) as f, open(output, "w") as fo:
        for lines in f.readlines():
            ele = lines.split()
            rnaseq = rnaseqdict[ele[0]]
            proseq = proseqdict[ele[1]]
            fo.writelines(f"{rnaseq},{proseq},{ele[2]}\n")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()