# -------------------------------------------------------------------
# this code reads bed files and select binding sites of about 101 nt
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import pandas as pd
import os, gzip
from subprocess import call

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Documents/RBP_CAMAS/data/"
source_path = f"{path}bedfiles_from_ENCORE/"  # raw downlods
base_file_path = f"{path}newdata/base_bed_files/"  # only around 101
overlap_file_path = f"{path}newdata/overlaps/"  # output of intersectBed
done_proteins_list = []

MIN = 95
MAX = 102

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def check_lenegth(startnum, endnum):
    fraglength = float(endnum) - float(startnum) + 1
    if MIN < fraglength < MAX:
        return True
    else:
        return False


def makedir_if_not_exist(path_to_check):
    if not os.path.exists(path_to_check):
        os.mkdir(path_to_check)


def find_overlaps(chromo):
    done_proteins_list = []
    for files in os.listdir(f"{base_file_path}{chromo}"):  # per 1 base file (1 protein and 1 chromosome)

        protein_name = files.replace(".bed", "")
        if len(protein_name) < 2:
            continue
        if "_2" in protein_name:
            protein_name = protein_name.replace("_2", "")
        if ".DS" in files or ".gz" in files:
            continue

        inputfile = f"{base_file_path}{chromo}/{files}"
        outputpath = f"{overlap_file_path}/{chromo}/{protein_name}/"
        makedir_if_not_exist(f"{overlap_file_path}/{chromo}/")
        makedir_if_not_exist(f"{overlap_file_path}/{chromo}/{protein_name}/")
        print(inputfile)
        for targetfile in os.listdir(source_path):  # per one target file
            if ".gz" in targetfile:
                continue
            print(f"search for overlaps between {protein_name} and {targetfile}..")
            if protein_name not in done_proteins_list:
                done_proteins_list.append(protein_name)
                command = f"intersectBed -a {inputfile} -b {source_path}{targetfile} -wao > {outputpath}{targetfile}"
            else:
                command = f"intersectBed -a {inputfile} -b {source_path}{targetfile} -wao >> {outputpath}{targetfile}"
            call([command], shell=True)


def make_base_files(targetchr):
    if not os.path.exists(f"{base_file_path}{targetchr}/"):
        print("###################3")
        os.mkdir(f"{base_file_path}{targetchr}/")
    for filenames in os.listdir(source_path):  # per 1 bed file, almost per protein
        if ".gz" in filenames:
            continue
        else:
            if ".bed" not in filenames:
                continue
            protein_name = ""
            # chr14_GL000194v1_random	72393	72489	GEMIN5_K562_rep0 ...
            with open(f"{source_path}{filenames}") as f:
                for lines in f.readlines():
                    ele = lines.split()
                    protein_name = ele[3].split("_")[0]
                    if protein_name in done_proteins_list:
                        protein_name += "_2"
                    else:
                        done_proteins_list.append(protein_name)
                    break
            with open(f"{source_path}{filenames}") as f, open(f"{base_file_path}{targetchr}/{protein_name}.bed", "w") as fo:
                for lines in f.readlines():
                    ele = lines.split()
                    if protein_name == "":
                        protein_name = ele[3].split("_")[0]
                    if check_lenegth(ele[1], ele[2]):  # if length close to 101
                        fo.writelines(lines)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # for i in range(1, 2):
    #     print(f"------------ CHROMOSOME {i}")
    #     make_base_files(f"chr{i}")
    for i in range(1, 2):
        print(f"------------ CHROMOSOME {i}")
        find_overlaps(f"chr{i}")
