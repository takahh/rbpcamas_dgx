# making mask data for RNA for benchmarks

def main():
    path2 = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/cross_attention/"

    tape_and_dna_aug = f"{path2}cr_8_2dsm.e10260134"
    tape_and_dna_no_aug = f"{path2}cr_8_1dsm_3-2.e10260151"
    no_tape_no_dnabert_aug = f"{path2}cr_8_1dsm.e10260147"
    no_tape_no_dnabert_no_aug = f"{path2}cr_8_1dsm.e10260147"
    path_list = [tape_and_dna_aug, tape_and_dna_no_aug, no_tape_no_dnabert_aug, no_tape_no_dnabert_no_aug]
    import matplotlib.pyplot as plt
    import numpy as np
    for index, inp in enumerate(path_list):
        alllist = []
        plt.figure()
        small_list = []
        name = inp.split("/")[-1].split(".")[0]
        with open(inp) as f:
            for lines in f.readlines():
                if "EPOCH" in lines:
                    alllist.append(small_list)
                    small_list = []
                elif "AUROC" in lines:
                    small_list.append(float(lines.split("[")[1].split("]")[0]))
            alllist.append(small_list)
        print(len(alllist))
        for idx, slist in enumerate(alllist):
            plt.plot(slist, label=name)
        plt.title(path_list[index].split("/")[-2:])
        plt.ylim(0, 2)
        # plt.hlines(y=0.69, xmin=0, xmax=600)
        # plt.plot(range(len(small_list), small_list))
        plt.show()


if __name__ == "__main__":
    main()
