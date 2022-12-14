# -------------------------------------------------------------------
# this code takes RNA and extend to be 101 nt
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import json
import os
import webbrowser
import requests

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
refseq = "/Users/mac/Documents/RBP_CAMAS/data/newdata/GRCh38_latest_genomic.fna"
rangelist = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq20_25_with_negatives.csv"
opath101 = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq20_25_with_negatives_all101.csv"
opath_w_sequences = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq20_25_with_negatives_RNAseq.csv"
bed = "/Users/mac/Documents/RBP_CAMAS/data/newdata/base_bed_files/chr1/AGGF1.bed"
fasta_file_for_mmseqs2 = "/Users/mac/Documents/RBP_CAMAS/data/newdata/fasta_for_mmseqs2.fa"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_sequence(chromo, start, end):
    api_url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chromo};start={start};end={end}"
    response = requests.get(api_url)
    try:
        json_response = json.loads(response.text)["dna"]
    except KeyError:
        print(api_url)
        return 0
    return json_response


def fill_to_101():
    with open(rangelist) as f, open(opath101, "w") as fo, open(fasta_file_for_mmseqs2, "w") as fasf:
        for lines in f.readlines():
            ele = lines.split(",")
            ele2 = ele[1].split("-")
            chrnum = ele2[0]
            start = int(ele2[1])
            end = int(ele2[2])
            # calculate the short length to 101 nt
            gap = 101 - (end - start)
            if gap > 0:
                left = gap // 2
                right = gap // 2 + gap % 2
                new_start = start - left
                new_end = end + right
                print(f"gap {gap}, and start from {start} to {new_start}, end from {end} to {new_end}")
                new_string = f"{ele2[0]}-{new_start}-{new_end}"
            else:
                new_string = ele[1]
            ele[1] = new_string
            sequence = get_sequence(chrnum, new_start, new_end)
            if sequence == 0:
                continue
            else:
                sequence = sequence.upper().replace("T", "U")
            fasf.writelines(f">{new_string}\n")
            fasf.writelines(f"{sequence}\n")
            # print(",".join(map(str, ele)))
            newline = sequence + "," + "".join(map(str, ele))
            fo.writelines(newline)


# def mmseqs2():


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    fill_to_101()
    # mmaseq2()