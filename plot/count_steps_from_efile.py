# -------------------------------------------------------------------
# this code count epochs so far for each models

# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/"
logpath = f"{path}single_node/"
logpath2 = f"{path}only_lncRNA/"
logpath3 = f"{path}unknown/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------

def count_steps(dirpath):
    count = 0
    for files in os.listdir(dirpath):
        if ".e" in files:
            with open(f"{dirpath}{files}") as f:
                mode = 0
                for lines in f.readlines():
                    if mode == 0:
                        if "train ROC" in lines:
                            mode = 1
                    elif mode == 1:
                        if "test ROC" in lines:
                            mode = 0
                            count += 1
    return count


def main():
    for dir in [logpath, logpath2, logpath3]:
        if "single" in dir:
            for dire in os.listdir(dir):
                if "past" in dire:
                    continue
                count = count_steps(f"{dir}/{dire}/")
                print(f"{dire} - {count}")
        count = count_steps(f"{dir}")
        print(f"{dir} - {count}")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()
