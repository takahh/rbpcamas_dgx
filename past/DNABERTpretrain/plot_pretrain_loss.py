# -------------------------------------------------------------------
# this code plots losses in pretraining
input = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/DNABERT/examples/pretrain_loss1.csv"
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    loss_list = []
    with open(input) as f:
        loss_list = [float(lines.split("(")[1].split(",")[0]) for lines in f.readlines()]
    plt.figure()
    xticks = numpy.arange(len(loss_list))
    plt.scatter(xticks, loss_list)
    plt.show()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()