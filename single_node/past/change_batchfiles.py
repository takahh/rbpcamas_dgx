# -------------------------------------------------------------------
# this code changes flags in batch files
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os, shutil
from subprocess import call

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/single_node/"
tpath = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/batchfiles/single_node/"

file_dict = {'no_tape_au': 'no_tape_aug',
    'lnc_tape_a': 'only_lncRNA',
    'tape_aug_a': 'tape_aug_augmultiply',
    'tape_aug_c': 'tape_aug_clip',
    'tape_aug_n': 'tape_aug_no_2dsfot',
    'tape_aug_r': 'tape_aug_only_rna',
    'tape_aug_p': 'tape_aug_only_pro',
    'tape_no_au': 'tape_no_aug',
    'no_tape_no': 'no_tape_no_aug',
    'tape_aug': 'tape_aug'}

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def correct_contents(fpath, mode, key=None):
    with open(fpath) as f, open(f"{fpath}.tmp", "w") as fo:
        for lines in f.readlines():
            if mode == 1:
                if "_node=" in lines:  # #$ -l f_node=1
                    fo.writelines(lines.replace("q_node", "f_node"))
                elif "node_name" in lines:  # -node_name f
                    fo.writelines(lines.replace(" q", " f"))
                elif "usechpoint" in lines:
                    fo.writelines(lines.replace(" 1", " 0"))
                elif lines[0:2] == "# ":
                    continue
                #################
                elif "#$ -hold_jid" in lines:
                    pass
                #################
                else:
                    fo.writelines(lines)
            elif mode == 2:  # for dependant jon submission
                if "#$ -p -5" in lines:
                    fo.writelines(f"#$ -p -5\n#$ -hold_jid {jobid_dict[key]}\n")
                # if "#$ hold_jid" in lines:
                #     fo.writelines(f"#$ -hold_jid {jobid_dict[key]}\n")
                    # fo.writelines(f"\n")
                else:
                    fo.writelines(lines)


def search_and_change(dir, mode):
    key = None
    if mode == 2:
        key = dir
        dir = f"{tpath}{file_dict[key]}"
    batchfile = ""
    for files in os.listdir(dir):
        if "test_" in files and "tmp" not in files:
            batchfile = f"{dir}/{files}"
            shutil.copy2(batchfile, f"{batchfile}.tmp")
            correct_contents(batchfile, mode, key)
            shutil.copy2(f"{batchfile}.tmp", batchfile)
            os.remove(f"{batchfile}.tmp")


def change():
    for dir in os.listdir(path):
        if "past" not in dir and "zip" not in dir:
            search_and_change(f"{path}{dir}", 1)
    # search_and_change(path2)


def get_job_ids():
    id_dict = {}
    with open("qstat.log") as f:
        for lines in f.readlines():
            if "2022" in lines:
                ele = lines.split()
                id_dict[ele[2]] = ele[0]
    return id_dict


def update_dependant_jobid():
    # call("ssh -t -t 18D38035@login.t3.gsic.titech.ac.jp -i ~/.ssh/id_rsa", shell=True)
    call("qstat>qstat.log", shell=True)
    global jobid_dict
    jobid_dict = get_job_ids()
    for key, item in jobid_dict.items():
        print(f"{key} {item}")
        search_and_change(key, 2)


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    change()
    # update_dependant_jobid()
