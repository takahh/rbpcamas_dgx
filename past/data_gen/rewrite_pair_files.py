# -------------------------------------------------------------------
# this code corrects mixed order of RNA and protein in pair files
import os

inp_path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/benchmarks/label_random_order/"  # RPI369_pairs.txt"
out_path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/benchmarks/label/"
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def check_if_RNA(pair_file_name, chain_name):
    #  RPI369_rna_seq.fa   RPI369_pairs.txt
    rna_file_name = pair_file_name.replace("pairs.txt", "rna_seq.fa")
    rna_file_name = rna_file_name.replace("label_random_order", "sequence")
    with open(rna_file_name) as f:
        for lines in f.readlines():
            if chain_name in lines:
                print("found!!")
                return True
    return False


def main():
    for files in os.listdir(inp_path):
        if "readme" in files or "_pos_" in files:
            continue
        if "369" not in files:
            continue
        else:
            pair_file_path = f"{inp_path}{files}"
            corrected_file_path = f"{out_path}{files}"
            with open(corrected_file_path, "w") as fo:
                with open(pair_file_path) as f:
                    for lines in f.readlines():
                        ele = lines.split("\t")
                        # if the first element is RNA, change the order
                        if check_if_RNA(pair_file_path, ele[0]):
                            fo.writelines(f"{ele[1]}\t{ele[0]}\t{ele[2]}")
                        else:
                            fo.writelines(lines)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()