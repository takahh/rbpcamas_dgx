# -------------------------------------------------------------------
# this code 
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import pysam
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
bamfile1 = "/Users/mac/Downloads/ENCFF730OHC.bam"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    bamfile = pysam.AlignmentFile(bamfile1, "rb")

    for read in bamfile:
        print(read)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()