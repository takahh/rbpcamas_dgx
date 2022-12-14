# -------------------------------------------------------------------
# this code decides five protein groups considering the file size of
# per-protein-files
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/selected_npy_per_RNA/inputdata/"  # DDX51positivefa_rep_seq.fasta.npy
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    # collect file size per protein
    protein_dict = {}
    for files in os.listdir(path):
        # DDX51positivefa_rep_seq.fasta.npy
        if "positive" in files:
            proname = files.split("positive")[0]
        elif "negative" in files:
            proname = files.split("negative")[0]
        filesize = os.path.getsize(f"{path}{files}")
        if proname in protein_dict.keys():
            protein_dict[proname] += float(filesize)
        else:
            protein_dict[proname] = float(filesize)

    # group by the file size
    dict_sorted = dict(sorted(protein_dict.items(), key=lambda item: item[1]))
    protein_groups = [[], [], [], [], []]
    for idx, (key, value) in enumerate(dict_sorted.items()):
        protein_groups[idx % 5].append(key)
    print(protein_groups)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()