# -------------------------------------------------------------------
# this code checks maximum RNA length in benchmarks
# input :
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
import os

path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/benchmarks/sequence/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------
overflie_list = []


def get_max_seq_len(filepath):
    maxlen = 0
    with open(filepath) as f:
        for lines in f.readlines():
            if ">" not in lines:
                linelength = len(lines.strip())
                if linelength > 2990:
                    if "rna" in filepath:
                        print(linelength)
                    if filepath not in overflie_list:
                        overflie_list.append(filepath)
                        print(filepath)
                if linelength > maxlen:
                    maxlen = linelength
    return maxlen


def main():
    pro_dict, rna_dict = {}, {}
    pre_max_rna_len, pre_max_pro_len = 0, 0
    for dir in os.listdir(path):
        if "rna" in dir:
            maxrnalen = get_max_seq_len(f"{path}{dir}")
            rna_dict[dir] = maxrnalen
            if maxrnalen > pre_max_rna_len:
                pre_max_rna_len = maxrnalen
        if "protein" in dir:
            maxprolen = get_max_seq_len(f"{path}{dir}")
            pro_dict[dir] = maxprolen
            if maxprolen > pre_max_pro_len:
                pre_max_pro_len = maxprolen
    print(f"max RNA is {pre_max_rna_len}, and max protein len is {pre_max_pro_len}")
    print(rna_dict)
    print(pro_dict)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()