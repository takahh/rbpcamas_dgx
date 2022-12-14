# -------------------------------------------------------------------
# this code checks data consistency
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# startdata = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/unknown_reduced_broad_share/0/0.npy"
# potentials = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/attn_arrays_hb_broad_share/0/0.npz"
# tokenized = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/training_data_tokenized_broad_share_red_8/0/0.npz"

startdata = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/known_protein_reduced/shared_4/0.npy"
potentials = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/attn_arrays_hb/shared_4/0.npz"
tokenized = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/training_data_tokenized/1400_shared_4/0.npz"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    star_arr = np.load(startdata, allow_pickle=True)
    print("Start")
    print(star_arr[0])

    pot_arr = np.load(potentials, allow_pickle=True)
    print("pot")
    print(pot_arr["pot"][0, :3, :5])

    tok_arr = np.load(tokenized, allow_pickle=True)
    print(tok_arr.files)
    print("protok")
    print(tok_arr["reduced_ptoks"].shape)
    print(tok_arr["reduced_ptoks"][:3, :5])
    print("rnatok")
    print(tok_arr["rnatok"][:3, :5])
    print("hbpots")
    print(tok_arr["hb_pots"].shape)
    print(tok_arr["hb_pots"][0, :3, :5])


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()