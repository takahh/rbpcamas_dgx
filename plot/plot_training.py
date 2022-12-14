# -------------------------------------------------------------------
# this code plot optimization for training
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import glob, os
import matplotlib.pyplot as plt
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/run_optimized/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    aucdict = {}
    for dir in glob.glob(f"{path}*.e*"):
        groupid = os.path.basename(dir).split(".")[0][-1]
        small_list = []
        with open(dir) as f:
            for lines in f.readlines():
                if "AUROC" in lines:
                    small_list.append(float(lines.split("AUROC:")[1].strip()))
        aucdict[groupid] = small_list
    plt.figure()
    plt.ylim(0.80, 0.94)
    plt.xlim(-1, 200)
    for key in aucdict.keys():
        plt.scatter(range(len(aucdict[key])), aucdict[key], label=key, s=5)
    plt.legend()
    plt.show()

# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()