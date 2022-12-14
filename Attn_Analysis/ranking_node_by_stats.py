# -------------------------------------------------------------------
# this code ranks nodes by stat pot for attn analysis
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
from past.cross_attn_101.prepare_att_aug import get_attn_dict
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
# pairid = "6V5B_C_D_from_head"
pairid = "6FF4_A_6_from_head"
# pairid = "6FF4_A_6_from_tail"
path = f"/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/data/pdbfiles/{pairid}/"
hbpath = f"{path}{pairid[:4]}_lean_with_node.hb2"
pipath = f"{path}{pairid[:4]}_snap_lean_with_node.txt"
id_dict = {"res": 1, "nuc": 3}
file_dict = {"hb": hbpath, "pi": pipath}
pot_dict_hb = get_attn_dict("hb")
pot_dict_pi = get_attn_dict("PI")
print(pot_dict_hb)
print(pot_dict_pi)

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def correct_letters(pairid):
    if "-" in pairid:
        pairid = pairid.upper()
        ele = pairid.split("-")
        pairid = f"{ele[1]}_{ele[0]}"
    return pairid


def get_pairlist(btype, node_id):  # res_hb, 50
    with open(file_dict[btype.split("_")[1]]) as f:
        if "res" in btype:
            pairlist = [x.split(",")[4] for x in f.read().split("\n") if len(x) > 0 and x.split(",")[1] == str(node_id)]
        else:
            pairlist = [x.split(",")[4] for x in f.read().split("\n") if len(x) > 0 and x.split(",")[3] == str(node_id)]
        return pairlist


def calc_sum(pair_list, bondtype):
    potsum = 0
    if "hb" in bondtype:
        for pair in pair_list:
            pair = correct_letters(pair)
            potsum += pot_dict_hb[pair]
    else:  # pi
        for pair in pair_list:
            pair = correct_letters(pair)
            potsum += pot_dict_pi[pair]
    return potsum


def calc_pot_sum(bondtype, idname):  # res_hb, 50
    pairlist = get_pairlist(bondtype, idname)
    if len(pairlist) > 0:
        potsum = calc_sum(pairlist, bondtype)
        return potsum
    else:
        return 0


def get_unique_list(datafile, indexnum):
    with open(datafile) as f:
        unique_list = list(set([x for x in f.read().split("\n") if len(x) > 0]))
        unique_list = list(set([int(x.split(",")[indexnum]) for x in unique_list]))
        unique_list = sorted(unique_list)
        return unique_list


def get_unique_lists(ntype, btype):
    unique_list = get_unique_list(file_dict[btype], id_dict[ntype])
    return unique_list


def main():
    nodeid_dict, potsum_dict = {}, {}

    # per node ranking

    for nodetype in ["res", "nuc"]:
        for bondtype in ["hb", "pi"]:
            nodeid_dict[f"{nodetype}_{bondtype}"] = get_unique_lists(nodetype, bondtype)

    for listtype in nodeid_dict.keys():
        potsum_dict = {}
        for nodeid in nodeid_dict[listtype]:
            potential_sum = calc_pot_sum(listtype, nodeid)
            potsum_dict[nodeid] = potential_sum
        potsum_dict = {k: potsum_dict[k] for k in sorted(potsum_dict, key=potsum_dict.get)}
        with open(f"{path}{listtype}_potranking_per_node.txt", "w") as f:
            for key, item in potsum_dict.items():
                f.writelines(f"{key},{item}\n")

    #  per interaction ranking
    with open(hbpath) as fh:
        hbpairlist = [f"HB,{x}" for x in fh.read().split("\n") if len(x) > 0]
    with open(pipath) as fp:
        pipairlist = [f"PI,{x}" for x in fp.read().split("\n") if len(x) > 0]
    print(pipairlist)

    def get_pair_score(pairlist, pot_dict):
        scorelist = []
        for item in pairlist:
            if item.split(",")[5][1] == "-":
                item = item.split(",")[5].replace("-", "_").upper()
                item = f"{item[2:]}_{item[0]}"
                print(item)
            else:
                item = item.split(",")[5]
            try:
                scorelist.append(pot_dict[item])
            except KeyError:
                print("keyerror")
                continue
        # scorelist = [pot_dict_hb[x.split(",")[5]] for x in pairlist]
        rank_dict = {}
        for item, score in zip(pairlist, scorelist):
            rank_dict[item] = score
        return rank_dict

    hb_score_dict = get_pair_score(hbpairlist, pot_dict_hb)
    pi_score_dict = get_pair_score(pipairlist, pot_dict_pi)
    print(pi_score_dict)

    hb_rank_dict = {k: hb_score_dict[k] for k in sorted(hb_score_dict, key=hb_score_dict.get, reverse=True)}
    pi_rank_dict = {k: pi_score_dict[k] for k in sorted(pi_score_dict, key=pi_score_dict.get, reverse=True)}

    with open(f"{path}hb_potranking_per_pair.txt", "w") as f:
        for key, item in hb_rank_dict.items():
            f.writelines(f"{key},{item}\n")
    with open(f"{path}pi_potranking_per_pair.txt", "w") as f:
        for key, item in pi_rank_dict.items():
            key = (key.replace("-", "_")).upper()
            f.writelines(f"{key},{item}\n")


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()