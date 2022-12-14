# -------------------------------------------------------------------
# this code 
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
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/LSTM/log.txt"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    train_loss_list, test_auc_list = [], []
    with open(path) as f:
        for lines in f.readlines():
            if "test" in lines and "loss" in lines:
                train_loss_list.append(float(lines.split()[7]))
                test_auc_list.append(float(lines.split()[13]))
    plt.figure()
    plt.plot(train_loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.show()
    plt.figure()
    plt.plot(test_auc_list)
    plt.ylabel("Test AUC")
    plt.xlabel("Epoch")
    plt.show()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()