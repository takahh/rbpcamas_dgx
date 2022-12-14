# -------------------------------------------------------------------
# this code inspect RPI7317 dataset
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
from matplotlib import pyplot as plt
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/LGFC-CNN/LGFC-CNN/Datasets/Train_dataset/NPinter_human/RPI7317.txt"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------

namelist = ["per RNA", "per protein"]

def main():
    for num in [0, 1]:
        itemlist = {}
        with open(path) as f:
            for lines in f.readlines():  # 1st column is lncRNA name
                ele = lines.split()
                if ele[num] not in itemlist:
                    itemlist[ele[num]] = 1
                else:
                    itemlist[ele[num]] += 1
        freqlist = list(itemlist.values())
        freqlist.sort()
        print(f"¥¥¥¥¥¥¥¥¥¥ {namelist[num]} ¥¥¥¥¥¥¥¥")
        print([key for key, value in itemlist.items() if value > 1000])
        print(f"unique count {len(freqlist)}")
        print(freqlist)
        print(freqlist.count(1))
        freqlist.sort(reverse=True)
        print(freqlist)
        plt.figure()
        plt.title(namelist[num])
        plt.hist(freqlist)
        plt.show()




# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()