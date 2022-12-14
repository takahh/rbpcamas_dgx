# -------------------------------------------------------------------
# this code 
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------
import pandas as pd
pd.set_option('display.max_columns', None)

def main():
    outpath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/run_optuna/result.csv"
    # paths = ["/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/run_optuna/optuna.o10761342"]
    paths = ["/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/run_optuna/optuna.o10776886"] # optuna.o10776458
    # paths = ["/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/batchfiles/run_optuna/optuna.o10776458"] # optuna.o10776458
    for path in paths:
        with open(path) as f:
            alllist, labellist = [], []
            firstline = True
            for lines in f.readlines():
                if "Best" not in lines:
                    continue
                print(lines)
                smalllist = []
                for item in lines.split(","):
                    if "Trial" in item:
                        smalllist.append(float(item.split(":")[1].split("-")[1].strip()))
                        # smalllist.append(int(item.split(":")[3].strip()))
                        if firstline:
                            labellist.append("loss")
                            # labellist.append("head_num")
                    # elif "Best" in item:
                        smalllist.append(int(item.split(":")[1].split("}")[0].strip()))
                        if firstline:
                            labellist.append("cross_dff")
                    # else:
                        smalllist.append(int(item.split(":")[1].strip()))
                        if firstline:
                            labellist.append(item.split(":")[0].strip(" ").strip("'"))
                alllist.append(smalllist)
                firstline = False
            df = pd.DataFrame(alllist)
            df.columns = labellist
            df.to_csv(outpath)
            df= df.sort_values(by="loss", ascending=False)
            print(df)

# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()