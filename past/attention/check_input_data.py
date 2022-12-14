# -------------------------------------------------------------------
# this code checks input data of potential
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
          'SER', 'THR', 'TRP', 'TYR', 'VAL']
baces = ['A', 'C', 'G', 'U']
aminos_pi = ["ARG", "TRP", "ASN", "HIS", "GLU", "GLN", "TYR", "PHE", "ASP"]

input1 = "/Users/mac/Desktop/t3_mnt/RNPopt/data/result/eval4/mean_pot_list/best_pot_subset2_nocv.csv"
input2 = "/Users/mac/Desktop/t3_mnt/RNPopt/data/result/eval4/optimized_normed_pot_list/best_pot_subset1_nocv.csv"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    df = pd.read_csv(input1, header=None)
    df2 = pd.read_csv(input2, header=None)
    plt.figure()
    # plt.scatter(np.arange(116), df.values[0])
    plt.scatter(np.arange(116), df2.values[0])
    plt.show()

# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()