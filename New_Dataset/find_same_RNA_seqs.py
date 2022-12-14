# -------------------------------------------------------------------
# this code finds the same RNA sequences in the original RBPsuite fasta files
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
from multiprocessing import Pool, Process

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------

PROCESS_COUNT = 12

run_mode = "T3"  # local or T3
run_mode = "local"  # local or T3
if run_mode == "local":
    path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/RBPsuite_data/"
    opath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/common_search_60000/"
elif run_mode == "T3":
    path = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/RBPsuite_data/"
    opath = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/common_search_60000/"
file1 = f"{path}AARSnegativefa_rep_seq.fasta"
file2 = f"{path}AATFpositivefa_rep_seq.fasta"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def write_to_file(file1, filename, seqlist):
    with open(f"{opath}/{file1}/{filename}", "w") as f:
        f.writelines("\n".join(seqlist))


def getseqlist(filepath):
    with open(f"{path}{filepath}") as f:
        seqlist = [x for x in f.read().split("\n") if ">" not in x]
        return seqlist


def compare_two_proteins(file1, file2):
    seqlist1 = getseqlist(file1)
    seqlist2 = getseqlist(file2)
    commonlist = [item1 for item1 in seqlist1 if item1 in seqlist2 and len(item1) > 2 and "NNNNNNNNNNNN" not in item1]
    write_to_file(file1, f"{file1}_{file2}", commonlist)


def get_filelistlist():
    # prepare file lists
    filelistall = [dirs for dirs in os.listdir(path) if "negative" in dirs or "positive" in dirs]
    filelist_list = []
    eachlen = int(len(filelistall) / PROCESS_COUNT)
    for i in range(PROCESS_COUNT):
        if i == PROCESS_COUNT - 1:
            filelist_list.append([filelistall[eachlen * i:]])
        else:
            filelist_list.append([filelistall[eachlen * i: eachlen * (i + 1)]])
    return filelist_list, filelistall


def search_files(pid):
    print(pid)
    filelistlist, filelist_all = get_filelistlist()
    filelist = filelistlist[pid][0]  # base list
    filelist2 = filelist_all       # target list
    for basefile in filelist:
        print(basefile)
        for targetfile in filelist2:
            if os.path.exists(f"{opath}/{basefile}/{basefile}_{targetfile}") or os.path.exists(f"{opath}/{basefile}/{targetfile}_{basefile}"):
                continue
            # skip if the pair is the same protein
            elif basefile.split("_")[0] == targetfile.split("_")[0]:
                continue
            else:
                compare_two_proteins(basefile, targetfile)


def my_err_cb(*args):
    print("error callback args={}".format(args))


def my_cb(*args):
    print("callback {}".format(args))


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    p = Pool(PROCESS_COUNT)
    result = p.map(search_files, range(PROCESS_COUNT))

