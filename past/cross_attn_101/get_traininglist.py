# -------------------------------------------------------------------
# this code get list of training and test
import os

path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
input = f"{path}RBPsuite_data/"
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    prolist = list(set([dir.split(".")[0] for dir in os.listdir(input)]))
    length = len(prolist)
    print(length)
    length = len(list(set(prolist)))
    print(length)
    eachlen = length // 5 + 1
    print(eachlen)
    finallist = []
    for i in range(5):
        if i < 4:
            finallist.append(prolist[i * eachlen: (i + 1) * eachlen])
        else:
            finallist.append(prolist[i * eachlen:])
    print(finallist)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()