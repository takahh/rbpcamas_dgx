# -------------------------------------------------------------------
# this code transform TAPE data into reduced one
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import numpy as np

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------

REDUCE_LEVEL = 8

path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/past/protein/"
proseq_master = f"/Users/mac/Documents/transformer_tape_dnabert/data/proteinseqs.csv"

if not os.path.exists(f"{path}red{REDUCE_LEVEL}/"):
    os.mkdir(f"{path}red{REDUCE_LEVEL}")

one2three_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

reduce_resi_dict = {
    20: ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
          'SER', 'THR', 'TRP', 'TYR', 'VAL'],
    13: ['TYR', 'LEU', 'LYS', 'GLY', 'TRP', 'ILE', 'ARG', 'PHE', 'CYS', 'HIS', 'ASP', 'ALA', 'ASN'],
    8: ['TYR', 'GLY', 'ILE', 'ARG', 'ASN', 'PHE', 'HIS', 'LYS'],
    4: ['LYS', 'ARG', 'PHE', 'ILE']
}

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_proseq(proname):
    sequence = None
    with open(proseq_master) as f:
        for lines in f.readlines():
            if proname == lines.split(",")[0]:
                sequence = lines.split(",")[1].strip()
    return sequence


def get_reduce_index(protein_seq):
    target_index = [idx for idx, x in enumerate(protein_seq) if one2three_dict[x] in reduce_resi_dict[REDUCE_LEVEL]]

    return target_index


def main():
    for files in os.listdir(path):
        if ".npy" not in files:
            continue
        # get protein seq from sequence
        pro_seq = get_proseq(files[:-4])
        # get index of the sequence
        red_idx = get_reduce_index(pro_seq)
        # load TAPE feature vectors
        tape_arr = np.load(f"{path}/{files}")
        if "AATF." in files:
            print(red_idx[-4:])
            print(len(pro_seq))
            print(tape_arr.shape)
        # reduce the TAPE feature vecs
        try:
            tape_arr_reduced = np.take(tape_arr, red_idx, axis=1)
        except IndexError:
            print(files)
            print(red_idx)
            print(tape_arr.shape)
        # write the reduced vectors
        np.save(f"{path}red{REDUCE_LEVEL}/{files}", tape_arr_reduced)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()