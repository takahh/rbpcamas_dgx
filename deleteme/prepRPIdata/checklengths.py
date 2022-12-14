# -------------------------------------------------------------------
# this code checks length of RNA and protein sequences
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/myRPI/RPI7317/"
input = f"{path}allseqpairs.csv"
output_rna = f"{path}rna_length_hist.png"
output_pro = f"{path}pro_length_hist.png"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def plothist(datalist, plotname, figpath):
    plt.figure()
    plt.title(plotname)
    plt.ylim(0, 200)
    plt.hist(datalist, bins=100)
    plt.savefig(figpath)
    plt.show()


def main():
    rnalengthlist, prolengthlist = [], []
    with open(input) as f:
        for lines in f.readlines():
            ele = lines[1:].replace("]", "").strip().split(",")
            rnalengthlist.append(len(ele[0]))
            prolengthlist.append(len(ele[1]))
    rnalisst101 = [x for x in rnalengthlist if x < 102]
    print(f"101 count {len(rnalisst101)}")
    rnalengthlist = sorted(rnalengthlist, reverse=True)
    prolengthlist = sorted(prolengthlist, reverse=True)

    print(f"max RNA length : {max(rnalengthlist)}, min {min(rnalengthlist)}, average {sum(rnalengthlist)/len(rnalengthlist)}")
    print(rnalengthlist)
    print(f"max protein length : {max(prolengthlist)}, min {min(prolengthlist)}")
    print(prolengthlist)
    plothist(rnalengthlist, "RNA", output_rna)
    plothist(prolengthlist, "protein", output_pro)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()