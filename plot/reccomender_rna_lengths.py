# -------------------------------------------------------------------
# this code checks RNA lengths in RNArecommender
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Documents/transformer_tape_dnabert/RNAcommender/examples/utrs.fa"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    lengthlist = []
    with open(path) as f:
        for lines in f.readlines():
            if ">" not in lines:
                lengthlist.append(len(lines.strip()))
    shortlist = [x for x in lengthlist if x < 102]
    print(len(shortlist))
    print(len(lengthlist))
    plt.figure()
    plt.hist(shortlist, bins=100)
    plt.show()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()