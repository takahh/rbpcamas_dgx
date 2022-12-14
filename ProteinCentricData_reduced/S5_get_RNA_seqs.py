# -------------------------------------------------------------------
# this code takes RNA and extend to be 101 nt
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import json, os
from subprocess import call
import requests
import time

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
refseq = "/Users/mac/Documents/RBP_CAMAS/data/newdata/GRCh38_latest_genomic.fna"
# original input file
rangelist = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq25_32_with_negatives.csv"

# fasta input for mmseqs2
fasta_file_for_mmseqs2 = "/Users/mac/Documents/RBP_CAMAS/data/newdata/fasta_for_mmseqs2.fa"
# result of mmseqs2
mmseq_reps = "/Users/mac/Documents/RBP_CAMAS/data/newdata/result_rep_seq.fasta"

# result before mmseqs2 application
opath101 = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq25_32_with_negatives_all101_tmp.csv"
# final result after mmseqs2 application
opath101_selected = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq25_32_with_negatives_all101.csv"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_sequence(chromo, start, end):
    api_url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chromo};start={start};end={end}"
    response = requests.get(api_url)
    try:
        json_response = json.loads(response.text)["dna"]
    except KeyError:
        # print(api_url)
        return 0
    return json_response


def fill_to_101():
    with open(rangelist) as f, open(opath101, "w") as fo, open(fasta_file_for_mmseqs2, "w") as fasf:
        for idx, lines in enumerate(f.readlines()):
            # if idx < 43177:
            #     continue
            # print(idx)
            ele = lines.split(",")
            ele2 = ele[1].split("-")
            chrnum = ele2[0]
            start = int(ele2[1])
            end = int(ele2[2])
            # if end - start + 1 > 101:
            #     print(ele)
            # calculate the short length to 101 nt
            gap = 101 - (end - start + 1)
            if gap != 0:
                left = gap // 2
                right = gap // 2 + gap % 2
                new_start = start - left - 1
                new_end = end + right
            else:
                new_start = start - 1
                new_end = end
            new_string = f"{ele2[0]}-{new_start}-{new_end}"
            # ele[1] = new_string
            sequence = get_sequence(chrnum, new_start, new_end)
            try:
                if len(sequence) != 101:
                    pass
                    # print(f"old query {start} {end}")
                    # print(f"query {new_string}, len {len(sequence)}")
            except Exception as e:
                print(e)
                continue
            else:
                sequence = sequence.upper().replace("T", "U")
            fasf.writelines(f">{new_string}\n")
            fasf.writelines(f"{sequence}\n")
            # print(",".join(map(str, ele)))
            newline = sequence + "," + "".join(map(str, ele))
            fo.writelines(newline)


def mmaseq2():
    os.chdir("/Users/mac/Documents/RBP_CAMAS/data/newdata")
    call([f"mmseqs easy-cluster fasta_for_mmseqs2.fa result tmp"], shell=True)


def apply_mmseq_results():
    with open(mmseq_reps) as mf, open(opath101) as f, open(opath101_selected, "w") as fo:
        mmseq_result_selected_rnas = [x.strip() for x in mf.readlines() if ">" not in x]
        print(len(mmseq_result_selected_rnas))
        for item in f.readlines():
            if item.split(",")[0] in mmseq_result_selected_rnas:
                print(item)
                fo.writelines(item)


def runs5():
    fill_to_101()
    mmaseq2()
    time.sleep(10)
    apply_mmseq_results()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    fill_to_101()
    mmaseq2()
    apply_mmseq_results()