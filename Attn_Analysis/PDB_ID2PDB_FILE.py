# -------------------------------------------------------------------
# this code choose pairs of RNA-protein where RNA length is more and around 101
# from the search result at PDB with "RNA and protein included as molecule type"
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import subprocess, os
import sys
sys.path.append('/Applications/PyMOL.app/Contents/lib/python3.7/site-packages')
import pymol
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/data/"
PATH = f"{path}pdbfiles/"
d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

array_out_path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/physical_interaction_map/"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def generate_pdb_file(chinfo):
    # get sequence list of the PDBID
    pdbid = chinfo[0]
    pymol.cmd.fetch(pdbid)
    fastastr = pymol.cmd.get_fastastr('all')
    pdbfilesdir = f"{PATH}/{chinfo[0]}_{chinfo[1]}_{chinfo[2]}/"
    with open(f"{pdbfilesdir}{pdbid}.fasta", "w") as f:
        f.writelines(fastastr)
    pymol.cmd.save(f"{pdbfilesdir}{pdbid}.pdb")
    pymol.cmd.reinitialize()


def run_hbplus(chinfo):
    id = chinfo[0]
    # change directory
    pdbfilesdir = f"{PATH}/{chinfo[0]}_{chinfo[1]}_{chinfo[2]}/"
    os.chdir(pdbfilesdir)

    # setup command strings
    hbplus_dir = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/hbplus"
    hbplus_cmd = f"{hbplus_dir}/hbplus"
    clean_cmd = f"{hbplus_dir}/clean"
    cmd1 = f"echo {id}.pdb|{clean_cmd}"
    cmd2 = f"{hbplus_cmd} {id}.pdb"
    cmd3 = f"x3dna-snap -i={id}.pdb -o={id}_snap.txt"

    # run clean
    subprocess.call(cmd1, shell=True)
    # run hbplus
    subprocess.call(cmd2, shell=True)
    # run SNAP
    subprocess.call(cmd3, shell=True)


def clip_output(chinfo):
    pchain = chinfo[1]
    rchain = chinfo[2]
    pdb_id = chinfo[0]
    pdbfilesdir = f"{PATH}/{chinfo[0]}_{chinfo[1]}_{chinfo[2]}/"
    hbout = f"{pdbfilesdir}{pdb_id}.hb2"
    piout = f"{pdbfilesdir}{pdb_id}_snap.txt"
    hbout_lean = f"{pdbfilesdir}{pdb_id}_lean.hb2"
    piout_lean = f"{pdbfilesdir}{pdb_id}_snap_lean.txt"
    hbout_only_rp_lean = f"{pdbfilesdir}{pdb_id}_lean_rp.hb2"  # related to RNA
    piout_only_rp_lean = f"{pdbfilesdir}{pdb_id}_snap_lean_rp.txt"
    hbout_only_rp_lean_perpro = f"{pdbfilesdir}{pdb_id}_lean_rp_perpro.hb2"  # related to protein
    piout_only_rp_lean_perpro = f"{pdbfilesdir}{pdb_id}_snap_lean_rp_perpro.txt"
    hbout_lean_with_node = f"{pdbfilesdir}{pdb_id}_lean_with_node.hb2"  # hbplus output
    piout_lean_with_node = f"{pdbfilesdir}{pdb_id}_snap_lean_with_node.txt"  # hbplus output

    # hbplus
    with open(hbout) as f, open(hbout_lean, "w") as fo, open(hbout_only_rp_lean, "w") as rp1, open(hbout_lean_with_node, "w") as fn\
            , open(hbout_only_rp_lean_perpro, "w") as fp:
        for lines in f.readlines():
            #           1111111111222222
            # 012345678901234567890123456
            #  ****          ****
            # *     ***     *     ***
            # G0201-LYS NZ  20124-  U OP1 2.78 SH  -2 -1.
            if lines[5] != "-":
                continue
            prior = lines[0]
            posterior = lines[14]
            truepair = f"{pchain}{rchain}"
            chpair1 = f"{prior}{posterior}"
            chpair2 = f"{posterior}{prior}"
            node1 = lines[6:9].strip()
            node2 = lines[20:23].strip()
            if len(node1 + node2) == 4:
                if prior == pchain:
                    res_locus = int(lines[1:5])
                    nuc_locus = int(lines[15:19])
                    res = prior
                    nuc = posterior
                    fp.writelines(f"{res},{res_locus},{nuc},{nuc_locus}\n")

                if posterior == rchain:
                    res_locus = int(lines[1:5])
                    nuc_locus = int(lines[15:19])
                    res = prior
                    nuc = posterior
                    fp.writelines(f"{res},{res_locus},{nuc},{nuc_locus}\n")

                if prior == rchain:
                    nuc_locus = int(lines[1:5])
                    res_locus = int(lines[15:19])
                    nuc = prior
                    res = posterior
                    nuc_letter = node1
                    res_letter = node2
                elif posterior == rchain:
                    res_locus = int(lines[1:5])
                    nuc_locus = int(lines[15:19])
                    res = prior
                    nuc = posterior
                    nuc_letter = node2
                    res_letter = node1
                else:
                    continue
                if truepair == chpair1 or truepair == chpair2:
                    fn.writelines(f"{res},{res_locus},{nuc},{nuc_locus},{res_letter}_{nuc_letter}\n")
                    fo.writelines(f"{res},{res_locus},{nuc},{nuc_locus}\n")
                if nuc == rchain and res != rchain:
                    rp1.writelines(f"{res},{res_locus},{nuc},{nuc_locus}\n")

    # SNAP
    seelines, parselines, ploc, rloc = 0, 0, 0, 0
    with open(piout) as f, open(piout_lean, "w") as fo, open(piout_only_rp_lean, "w") as rp1, open(piout_lean_with_node, "w") as fn,\
            open(piout_only_rp_lean_perpro, "w") as fp:
        for lines in f.readlines():
            # List of 81 base/amino-acid stacks
            #        id   nt-aa   nt           aa      vertical-distance   plane-angle
            #    1  6ZLW  A-tyr  2.A2         D.TYR223        3.70             26
            #    2  6ZLW  G-gln  2.G10        D.GLN115        4.01             36
            #    3  6ZLW  G-arg  2.G71        G.ARG170        3.68             56
            if seelines == 0:
                if "amino-acid stacks" in lines:
                    seelines = 1
            else:
                if parselines == 0:
                    if "vertical-distance" in lines:
                        parselines = 1
                else:
                    # parse lines
                    if len(lines) < 3:
                        break
                    ele = lines.split()
                    ch1 = ele[3].split(".")[0]
                    ch2 = ele[4].split(".")[0]
                    ploc = ele[4].split(".")[1][3:]
                    rloc = ele[3].split(".")[1][1:]
                    rp1.writelines(f"{ch2},{ploc},{ch1},{rloc}\n")

                    if ch1 == pchain:
                        ploc = ele[3].split(".")[1][3:]
                        rloc = ele[4].split(".")[1][1:]
                        fp.writelines(f"{pchain},{ploc},{rchain},{rloc}\n")
                        if ch2 == rchain:
                            ploc = ele[3].split(".")[1][3:]
                            rloc = ele[4].split(".")[1][1:]
                            fo.writelines(f"{pchain},{ploc},{rchain},{rloc}\n")
                            fn.writelines(f"{pchain},{ploc},{rchain},{rloc},{ele[2]}\n")
                    elif ch2 == pchain:
                        ploc = ele[4].split(".")[1][3:]
                        rloc = ele[3].split(".")[1][1:]
                        fp.writelines(f"{pchain},{ploc},{rchain},{rloc}\n")
                        if ch1 == rchain and ch2 == pchain:
                            ploc = ele[4].split(".")[1][3:]
                            rloc = ele[3].split(".")[1][1:]
                            fo.writelines(f"{pchain},{ploc},{rchain},{rloc}\n")
                            fn.writelines(f"{pchain},{ploc},{rchain},{rloc},{ele[2]}\n")



def select_101(chaindata, bondtype):
    pdb_id = chaindata[0]
    pdbfilesdir = f"{PATH}/{chaindata[0]}_{chaindata[1]}_{chaindata[2]}/"
    if bondtype == "hb":
        leanfile = f"{pdbfilesdir}{pdb_id}_lean.hb2"  # hbplus output
        rponly_file = f"{pdbfilesdir}{pdb_id}_lean_rp.hb2"
    else:
        leanfile = f"{pdbfilesdir}{pdb_id}_snap_lean.txt"  # 5AN9_snap_lean.txt
        rponly_file = f"{pdbfilesdir}{pdb_id}_snap_lean_rp.txt"
        rponly_perpro_file = f"{pdbfilesdir}{pdb_id}_snap_lean_rp_per_protein.txt"
    # make a list of rna loci in contact
    with open(leanfile) as f:
        # SER,67,G,113
        rna_loc_list = [int(x.split(",")[3].strip()) for x in f.readlines()]
        try:
            max_loc = max(rna_loc_list)
            min_loc = min(rna_loc_list)
        except ValueError:
            return -1
        if bondtype == "hb":
            # move = (min_loc - 1)
            move = 1
        else:
            move = 1
        if min_loc > move:
            start_loc0 = min_loc - move
        else:
            start_loc0 = min_loc
    srart_loc0 = 1
    count_dict = {}
    pchain_contact_count = {}
    ecr_dict = {}
    buffer = 0
    if "6ID1" in pdb_id:
        pass
    #     if bondtype == "hb":
    #         buffer = 45
    #     else:
    #         buffer = 90
    elif "7Q4O" in pdb_id:
        if bondtype == "hb":
            buffer = 0
        else:
            buffer = 0
    # ---------------------------------------------------
    # make a dictionary(key: start locuss, value: count)
    # ---------------------------------------------------
    for i in range(100):  # per range
        start_loc = i * 101 + start_loc0
        # if "6ID1" in pdb_id:
        #     if len(rna_loc_list) != 1 and start_loc + 101 > 116:
        #         break
        with open(leanfile) as f:
            interaction_list = [x for x in f.readlines() if start_loc < int(x.split(",")[3].strip()) < start_loc + 101]
            count_dict[start_loc] = len(interaction_list)
        with open(rponly_file) as f:
            print(chaindata)
            print(rponly_file)
            # choose corresponding contacts with the same RNA chain ID
            interaction_list = [x for x in f.readlines() if start_loc < int(x.split(",")[3].strip()) < start_loc + 101
                                and x.split(",")[2] == chaindata[2] and x.split(",")[0] != chaindata[2]]

            pchain_contact_count[start_loc] = len(interaction_list)

    for key in count_dict.keys():
        target_count = count_dict[key]
        all_rp_count = pchain_contact_count[key]
        if target_count == 0:
            ecr_dict[key] = 0
        else:
            ecr_dict[key] = target_count/all_rp_count
    start_loc_best = max(ecr_dict, key=ecr_dict.get)
    return start_loc_best


def write_true_fasta_from_pdbfile(startnum, chdata, bondtype):
    pchain = chdata[1]
    rchain = chdata[2]
    pdb_id = chdata[0]
    data_id = f"{chdata[0]}_{chdata[1]}_{chdata[2]}"
    pdbfilesdir = f"{PATH}/{data_id}/"
    pdbfile = f"{pdbfilesdir}{pdb_id}.pdb"
    true_fasta = f"{pdbfilesdir}{pdb_id}_tru_fasta_{bondtype}.csv"
    # add_to_hb_pre = "1,G\n2,U\n3,G\n4,C\n5,U\n6,C\n7,G\n8,C\n9,U\n10,U\n11,C\n12,G\n13,G\n14,C\n15,A\n16,G\n17,C\n18," \
    #                 "A\n"  # GUGCUCGCUUCGGCAGCA
    # add_to_hb_post = "96,U\n97,U\n98,C\n99,C\n100,A\n101,U\n"
    # add_to_hb_post = "96,U\n97,U\n98,C\n99,C\n100,A\n101,U\n102,A\n103,U\n104,U\n105,U\n106,U\n107,U\n"  # for 6FF4_A_6
    # add_to_pi = "4,C\n5,U\n6,C\n7,U\n8,G\n9,G\n10,U\n11,U\n12,U\n13,C\n14,U\n15,C\n16,U\n17,U\n18,C\n19,A\n20,G\n21,A\n"

    if pdb_id == "6V5B":
        add_to_hb_pre = "1,C\n2,U\n"
        add_to_mid = "42,G\n43,U\n44,A\n45,G\n46,U\n47,G\n48,A\n49,A\n50,A\n51,U\n52,A\n53,U\n54,A\n55,U\n56,A\n57,U\n58,U\n59,A\n60,A\n61,A\n62,C\n"

    # RNA sequence
    with open(pdbfile) as f, open(true_fasta, "w") as fo:
        if bondtype == "hb":
            fo.writelines(add_to_hb_pre)
        # else:
        #     fo.writelines(add_to_pi)
        written_loc_list = []
        nuc_count = 0
        for lines in f.readlines():
            if "ATOM" not in lines[0:4]:
                continue
            # 0123456789012345678901234567890123456789
            #                    * * ***
            # ATOM   1524  OP1   C 2  72     168.785  77.260 164.728  1.00 47.06      A    O
            # ATOM   1525  OP2   C 2  72     167.854  78.724 162.853  1.00 47.06      A    O1-
            # ATOM   1526  P     U 2  76     149.319  73.120 168.125  1.00 42.88      A    P
            if lines[21] == rchain:
                locus = lines[22:26].strip()
                if int(locus) >= startnum:
                    if locus in written_loc_list:
                        continue
                    fo.writelines(f"{locus},{lines[19]}\n")
                    if pdb_id == "6V5B":
                        if int(locus) == 41:
                            fo.writelines(add_to_mid)
                    written_loc_list.append(locus)
                    nuc_count += 1
                    if nuc_count == 101:
                        break

        # if bondtype == "hb":
        #     fo.writelines(add_to_hb_post)
    # protein sequence
    with open(pdbfile) as f, open(true_fasta, "a") as fo:
        written_loc_list = []
        for lines in f.readlines():
            print(lines)
            if "ATOM" not in lines[0:4]:
                continue
            #           1111111111222222222233333333334444444444555
            # 0123456789012345678901234567890123456789012345678901234567890123456789
            #                  *** * ***
            # ATOM   1524  OP1   C 2  72     168.785  77.260 164.728  1.00 47.06      A    O
            if lines[21] == pchain:
                locus = lines[22:26].strip()
                if locus in written_loc_list:
                    continue
                print(f"{locus},{lines[17:20]}, {d[lines[17:20]]}\n")
                fo.writelines(f"{locus},{lines[17:20]}, {d[lines[17:20]]}\n")
                written_loc_list.append(locus)


def write_seq_for_model(startid, chdata, bondtype):
    pdb_id = chdata[0]
    pdbfilesdir = f"{PATH}/{chdata[0]}_{chdata[1]}_{chdata[2]}/"
    true_fasta = f"{pdbfilesdir}{pdb_id}_tru_fasta_{bondtype}.csv"
    fasta_for_model = f"{pdbfilesdir}{pdb_id}_fasta_for_model_{bondtype}.csv"
    rnaseq, proseq = "", ""
    with open(true_fasta) as f, open(fasta_for_model, "w") as fo:
        for lines in f.readlines():
            ele = lines.split(",")
            kw = ele[1].strip()
            print(f"{int(ele[0])} >= {startid}")
            print(f"{int(ele[0]) >= startid}")
            if int(ele[0]) >= startid:
                if len(kw) == 1:
                    rnaseq += kw
            if len(kw) == 3:
                proseq += ele[2].strip()
        fo.writelines(f"{proseq},{rnaseq}\n")


def write_nparray(chainfo, bondtype):
    pdb_id = chainfo[0]
    data_id = f"{chainfo[0]}_{chainfo[1]}_{chainfo[2]}"
    pdbfilesdir = f"{PATH}/{chainfo[0]}_{chainfo[1]}_{chainfo[2]}/"
    true_fasta = f"{pdbfilesdir}/{pdb_id}_tru_fasta_{bondtype}.csv"
    fasta_for_model = f"{pdbfilesdir}{pdb_id}_fasta_for_model_{bondtype}.csv"

    # get true lists
    with open(true_fasta) as f:
        rna_true_list = [x.split(",")[0] for x in f.readlines() if len(x.split(",")[1].strip()) == 1]
    with open(true_fasta) as f:
        pro_true_list = [x.split(",")[0] for x in f.readlines() if len(x.split(",")[1]) == 3]

    # write sequence for model
    with open(fasta_for_model) as f:
        contents = f.read()
        prolen = len(contents.split(",")[0])
        rnalen = len(contents.split(",")[1].strip())

    # sort contact info and put it into dict
    if bondtype == "hb":
        leanfile = f"{pdbfilesdir}{pdb_id}_lean.hb2"  # hbplus output
    else:
        leanfile = f"{pdbfilesdir}{pdb_id}_snap_lean.txt"  # 5AN9_snap_lean.txt
    with open(leanfile) as f:
        contactlist = f.readlines()  # SER,67,G,113 ...
    # make a zero array
    arr = np.zeros([prolen, rnalen])

    # add 1 according to interaction data
    for item in contactlist:
        try:
            rna_loc = item.split(",")[3].strip()
            pro_loc = item.split(",")[1]
            arr[pro_true_list.index(str(pro_loc)), rna_true_list.index(str(rna_loc))] += 1
        except ValueError:
            continue

    np.savez_compressed(f"{array_out_path}/{bondtype}_{data_id}", interactions=arr, rna_true_list=np.array(rna_true_list),
                        pro_true_list=np.array(pro_true_list))

    return arr, rna_true_list, pro_true_list


def heatmap(arr_data, chdata, bondtype, ploclist, rloclist):
    title = ""
    if f"{chdata[0]}_{chdata[1]}_{chdata[2]}" == "6ID1_A_B":
        pstart = 400
        pend = 650
        rend = 45
    else:
        pstart = 0
        pend = 2000
        rend = 1000
    if bondtype == "hb":
        title = f"{chdata[0]}_{chdata[1]}_{chdata[2]}, Hydrogen Bond\n"
    else:
        title = f"{chdata[0]}_{chdata[1]}_{chdata[2]}, Pi Interaction\n"
    plt.figure()
    plt.title(title)
    plt.gcf().subplots_adjust(bottom=0.15)
    df = pd.DataFrame(arr_data[pstart:pend, :rend], index=ploclist[pstart:pend], columns=rloclist[:rend])
    ax = seaborn.heatmap(df, vmin=0, vmax=np.max(arr_data), cmap="Greys", cbar=True)
    ax.tick_params(axis='both', which='both', length=0)
    plt.xlabel("RNA")
    plt.ylabel("Protein")
    plt.xlabel("RNA")
    plt.show()


def main(chain_info_list):
    for chain_info in chain_info_list:
        print(f"############ {chain_info}")
        for mode in ["hb", "pi"]:
            # if not os.path.exists(f"{PATH}/{chain_info[0]}_{chain_info[1]}_{chain_info[2]}"):
            #     os.mkdir(f"{PATH}/{chain_info[0]}_{chain_info[1]}_{chain_info[2]}")
            # # generate fasta and pdb files
            # generate_pdb_file(chain_info)
            # # run hbplus
            # run_hbplus(chain_info)
            # # remove res-res contacts and else
            # clip_output(chain_info)
            # decide best start position in RNA
            start_nuc_loc = 0
            # start_nuc_loc = select_101(chain_info, mode)
            # if start_nuc_loc < 0:
            #     break
            # print(f"{start_nuc_loc} start")
            if chain_info[0] == "6ICZ":
                start_nuc_loc = 7
            # write true fasta
            write_true_fasta_from_pdbfile(start_nuc_loc, chain_info, mode)
            # one line two sequence file for model use
            # write_seq_for_model(start_nuc_loc, chain_info, mode)
            # make an array for heatmap
            arr, ploc_list, rloc_list = write_nparray(chain_info, mode)
            # make a heatmap
            # heatmap(arr, chain_info, mode, rloc_list, ploc_list)
            print(f"{mode} heatmap done")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # chai_info_list = [["6ID1", "A", "B"]]
    # chai_info_list = [["6QDV", "A", "6"]]
    # chai_info_list = [["6ICZ", "A", "F"]]
    # chai_info_list = [["6ID1", "A", "F"]]
    # chai_info_list = [["6FF4", "A", "6"]]
    chai_info_list = [["6V5B", "C", "D"]]
    # chai_info_list = [["5AN9", "J", "N"], ["6ZLW", "F", "2"], ["6ZLW", "K", "2"], ["6ZXG", "L", "2"]]

    main(chai_info_list)
