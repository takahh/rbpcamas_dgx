# -------------------------------------------------------------------
# this code copies only tfevents at logs directory
# to avoid resource error in uploading to tb server
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
import os, shutil

path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/logs"
# /logs/node_fd...0_t3_20220427/train/events.out.tfevents.1651056653.r6i6n0.56666.788.v2   OK
# /logs/node_f_noden...atu.0427/train/events.out.tfevents.1651056668.r6i6n0.profile-empty   NO

opath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/logs_onlytfevnt/"
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def copy_tfevents(keyword):
    for dir in os.listdir(path):
        if os.path.exists(f"{path}/{dir}/{keyword}") is False:
            continue
        for dir2 in os.listdir(f"{path}/{dir}/{keyword}"):
            if "tfevent" in dir2 and "profile" not in dir2:
                file_to_copy = f"{path}/{dir}/{keyword}/{dir2}"
                destination1 = f"{path.replace('logs', 'logs_onlytfevnt')}/{dir}/"
                destination2 = f"{path.replace('logs', 'logs_onlytfevnt')}/{dir}/{keyword}/"
                if os.path.exists(destination1) is False:
                    os.mkdir(destination1)
                if os.path.exists(destination2) is False:
                    os.mkdir(destination2)
                try:
                    shutil.copy2(file_to_copy, destination2)
                except OSError:
                    print(dir)
                    print(dir2)


def main():
    if os.path.exists(opath) is False:
        os.mkdir(opath)
    copy_tfevents("train")
    copy_tfevents("validation")


# -------------------------------------------------------------------　
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()