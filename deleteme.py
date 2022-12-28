# -------------------------------------------------------------------
# this code runs sMMseqs2 to find common RNA
# input :
# output:
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os
import shutil
from subprocess import call
from multiprocessing import Pool
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
originpath = f"/Users/mac/Documents/transformer_tape_dnabert/data/clusteredRBPsuite/"
basepath = f"/Users/mac/Documents/transformer_tape_dnabert/data/clusteredRBPsuite/10000/"
input = f"{basepath}/"
output = f"{basepath}DB/"
simpleheadpath = f"{basepath}simple_header/"
PALARELL = 10
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------

    # os.chdir(opath)
    # for files in os.listdir(ipath):
    #     if os.path.isdir(f"{ipath}{files}"):
    #         continue
    #     call([f"mmseqs easy-linclust {ipath}{files} {files.replace('.', '')} tmp"], shell=True)
    #     os.remove(f"{files.replace('.', '')}_all_seqs.fasta")
    #     os.remove(f"{files.replace('.', '')}_cluster.tsv")
    #     call([f"rm -Rf ./tmp"], shell=True)
    #     # break

def make_fasta2db():
    for files in os.listdir(simpleheadpath):
        if "tmp" in files:
            continue
        if os.path.isdir(f"{simpleheadpath}{files}"):
            continue
        print(files)
        call([f"mmseqs createdb {simpleheadpath}{files} {output}{files}_DB"], shell=True)


def removeresultDB(tarname):
    for files in os.listdir(basepath):
        if f"{tarname}" in files:
            os.remove(f"{basepath}/{files}")


def make_smaller_summary(filepath):
    output = f"{filepath}_small"
    with open(filepath) as f, open(output, "w") as fo:
        for lines in f.readlines():
            ele = lines.split()
            if float(ele[4]) > 90 and float(ele[5]) > 0.9:
                fo.writelines(lines)
    os.remove(filepath)


def search(query, target):  # query : AARSnegative, target : AATFnegative
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    os.mkdir("tmp")
    queryDB = f"{output}{query}_DB"
    targetDB = f"{output}{target}_DB"
    targetname = f"{target.replace('fa_rep_seq.fasta', '')}"
    removeresultDB(targetname)

    # prepare folder
    if "negative" in query:
        proname = query.replace("negative", "").replace("fa_rep_seq.fasta", "")
        query_label = "nagative"
    elif "positive" in query:
        proname = query.replace("positive", "").replace("fa_rep_seq.fasta", "")
        query_label = "positive"
    print(f"proname is {proname} #####################################")
    if not os.path.exists(f"{basepath}result/{proname}"):
        os.mkdir(f"{basepath}result/{proname}")
    final_opath = f"{basepath}result/{proname}/{query_label}_{targetname}"
    final_output = f"{final_opath}summary.m8"
    if os.path.exists(final_output):
        pass
    else:
        # run mmseqs2
        call([f"mmseqs search --search-type 3 {queryDB} {targetDB} {targetname} tmp"], shell=True)
        # make readable summary
        call([f"mmseqs convertalis --search-type 3 --format-output 'qheader,qseq,theader,tseq,alnlen,fident,nident' {queryDB} {targetDB} {targetname} {targetname}.m8"], shell=True)
        shutil.copy2(f"{targetname}.m8", final_output)
        removeresultDB(targetname)
        shutil.rmtree("tmp")
        make_smaller_summary(final_output)


def get_file_list():
    file_list = []
    for files in os.listdir(input):
        if "tmp" in files or ".DS" in files:
            continue
        if os.path.isdir(f"{input}{files}"):
            continue
        print(files)
        file_list.append(files)
    return file_list


def fasta_header_simpler():
    opath = simpleheadpath
    count = 0
    for files in get_file_list():
        with open(f"{input}{files}") as f, open(f"{opath}{files}", "w") as fo:
            for lines in f.readlines():
                if ">" in lines:
                    fo.writelines(f">{count}\n")
                    count += 1
                else:
                    fo.writelines(lines)


def run_mmseqs2():
    pathlist = get_file_list()
    donelist = []
    for query_name in pathlist:
        donelist.append(query_name)
        for target_name in pathlist:
            if target_name in donelist:
                continue
            else:
                search(query_name, target_name)


def extract_10000_from_original():
    for files in os.listdir(originpath):
        if os.path.isdir(f"{originpath}/{files}") or ".DS" in files:
            continue
        with open(f"{originpath}/{files}") as f, open(f"{basepath}{files}", "w") as fo:
            fo.write("\n".join(f.read().split("\n")[:20000]))


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()

