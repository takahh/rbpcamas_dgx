# -------------------------------------------------------------------
# this code selects final pairs from the previous selections
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/PDBID_list_RNA_is_102_150.txt"
opath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/PDBID_list_RNA_is_102_119.txt"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    with open(path) as f, open(opath, "w") as fo:
        for lines in f.readlines():
            numlist = [int(x[:-1]) for x in lines.split() if ":" not in x]
            print(numlist)
            for num in numlist:
                if num < 120 and num > 101:
                    fo.writelines(lines)



# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()