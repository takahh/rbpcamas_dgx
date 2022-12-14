# -------------------------------------------------------------------
# this code mmakes batch files
import shutil

input = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/more_shared/"
output = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/batch_files/more_shared/"

# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    for i in range(10, 30):
        # -----------
        # batch files
        # -----------
        with open(f"{input}batch{i}.sh", "w") as fo:
            fo.writelines(f"qsub -g tga-science -m abe -M d@wakou.cc test{i}.sh")

        # -----------
        # test files
        # -----------
        with open(f"{input}test{i}.sh", "w") as fo, open(f"{input}test0.sh") as f:
            for lines in f.readlines():
                if "group0" in lines:
                    fo.writelines(lines.replace("group0", f"group{i}"))
                elif "--group" in lines:
                    fo.writelines(lines.replace("0", f"{i}"))
                else:
                    fo.writelines(lines)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()