# -------------------------------------------------------------------
# this code finds the protein that has the most 101 ranges
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
size_dict = {}
path = "/Users/mac/Documents/RBP_CAMAS/data/newdata/overlaps/chr1/"  # AARS/ENCFF005ZCI.bed"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    for proteins in os.listdir(path):
        if "." in proteins:
            continue
        sizes = []
        for files in os.listdir(f"{path}{proteins}"):
            filename = f"{path}{proteins}/{files}"
            sizes.append(os.path.getsize(filename))
        size_dict[proteins] = sum(sizes)/len(sizes)

    sorted_size_dict = {k: v for k, v in sorted(size_dict.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_size_dict)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()