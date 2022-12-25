# -------------------------------------------------------------------
# this code runs S1-8 script, and tar the final directory
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
from S1_find_hotspots import runs1
from S2_count_repeats import runs2
from S3_select20to1file import runs3
from S4_collect_negatives import runs4
from S5_get_RNA_seqs import runs5
from S6_proid_to_reducedseq import runs6
from S7_prepare_potential_tables import runs7
from S8_tokenizer import runs8

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# Min posi count per rna
LEAST_POSI_COUNT_PER_RNA = 2
# number of uniques rna
MAX_SITE_COUNT = 2 * (5000//(LEAST_POSI_COUNT_PER_RNA * 2))
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    # print("#### S1 start ####")
    # runs1()
    # print("#### S2 start ####")
    # runs2()
    # print("#### S3 start ####")
    # runs3(LEAST_POSI_COUNT_PER_RNA, MAX_SITE_COUNT)
    # print("#### S4 start ####")
    # runs4(LEAST_POSI_COUNT_PER_RNA)
    # print("#### S5 start ####")
    # runs5()
    # print("#### S6 start ####")
    # runs6(LEAST_POSI_COUNT_PER_RNA)
    # print("#### S7 start ####")
    # runs7()
    print("#### S8 start ####")
    runs8(LEAST_POSI_COUNT_PER_RNA)
    print("#### All Done ####")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()
