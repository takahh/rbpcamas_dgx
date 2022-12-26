# -------------------------------------------------------------------
# this code runs S1-6, N6-8 script, and tar the final directory,
# and copy to the "fumidai" server
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
from ProteinCentricData_reduced.S1_find_hotspots import runs1
from ProteinCentricData_reduced.S2_count_repeats import runs2
from ProteinCentricData_reduced.S3_select20to1file import runs3
from ProteinCentricData_reduced.S4_collect_negatives import runs4
from ProteinCentricData_reduced.S5_get_RNA_seqs import runs5
from N6_proid_to_reducedseq import runn6
from N7_prepare_potential_tables import runn7
from N8_tokenizer import runn8

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# Min posi count per rna
LEAST_POSI_COUNT_PER_RNA = 15
# number of uniques rna
MAX_SITE_COUNT = 2 * (5000//(LEAST_POSI_COUNT_PER_RNA * 2))
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    print("#### S1 start ####")
    runs1()
    print("#### S2 start ####")
    runs2()
    print("#### S3 start ####")
    runs3(LEAST_POSI_COUNT_PER_RNA, MAX_SITE_COUNT)
    print("#### S4 start ####")
    runs4(LEAST_POSI_COUNT_PER_RNA)
    print("#### S5 start ####")
    runs5()
    print("#### N6 start ####")
    runn6(LEAST_POSI_COUNT_PER_RNA)
    print("#### N7 start ####")
    runn7()
    print("#### N8 start ####")
    runn8(LEAST_POSI_COUNT_PER_RNA)
    print("#### All Done ####")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()
