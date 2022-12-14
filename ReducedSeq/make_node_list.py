# -------------------------------------------------------------------
# this code makes a node list depending on the threshold
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import pandas as pd
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
####################
THRESHOLD = 0
####################

aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
          'SER', 'THR', 'TRP', 'TYR', 'VAL']
baces = ['A', 'C', 'G', 'U']
aminos_pi = ["ARG", "TRP", "ASN", "HIS", "GLU", "GLN", "TYR", "PHE", "ASP"]
one2three_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
best_pot_normed = f"/Users/mac/Documents/T3_groupdisk_download_manual/RNPopt/RNPopt/data/result/eval4/optimized_normed_pot_list/best_pot_subset1_nocv.csv"

output = "/Users/mac/Desktop/t3_mnt/reduced_RBP_camas/data/strong_nodes/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_attn_dict(type):
    attn_list = pd.read_csv(best_pot_normed, header=None).values[0]
    attn_list = -attn_list
    # attn_list = softmax(-attn_list)
    attention_dict = {}
    if type == "PI":
        count = 80
        for amino in aminos_pi:
            for base in baces:
                attention_dict[f"{amino}_{base}"] = attn_list[count]
                count += 1
    else:
        count = 0
        for amino in aminos:
            for base in baces:
                attention_dict[f"{amino}_{base}"] = attn_list[count]
                count += 1
                if count == 80:
                    break
    return attention_dict


def main():
    attndict_hb = get_attn_dict("HB")
    attndict_pi = get_attn_dict("PI")
    print(attndict_hb)
    print(attndict_pi)
    highestpot = max(attndict_pi.values())
    print(highestpot)
    splits = 10
    for i in range(splits):
        node_list = []
        threshold = highestpot/splits * i
        print(threshold)
        for keys in attndict_hb.keys():
            if attndict_hb[keys] > threshold:
                node_list.append(keys.split("_")[0])
        for keys in attndict_pi.keys():
            if attndict_pi[keys] > threshold:
                node_list.append(keys.split("_")[0])
        nodelist = list(set(node_list))
        count = len(list(set(node_list)))
        print(f"i {i}, count {count}, list {nodelist}")
        with open(f"{output}{count}.csv", "w") as f:
            for item in nodelist:
                f.writelines(f"{item},")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()