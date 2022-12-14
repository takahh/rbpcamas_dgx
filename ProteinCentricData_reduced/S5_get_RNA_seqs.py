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
rangelist = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq25_32_with_negatives.csv"
opath101 = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq25_32_with_negatives_all101.csv"
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
            if end - start + 1 > 101:
                print(ele)
            # calculate the short length to 101 nt
            gap = 101 - (end - start + 1)
            if gap > 0:
                left = gap // 2
                right = gap // 2 + gap % 2
                new_start = start - left
                new_end = end + right
                new_string = f"{ele2[0]}-{new_start}-{new_end}"
                print(f"{new_string} extended")
            else:
                new_string = ele[1]
                new_start = start
                new_end = end
                print(lines)
                print(new_string)
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


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    fill_to_101()
    # mmaseq2()