# -------------------------------------------------------------------
# this code tokenize sequences
# input : sequence
# output: array of tokens
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import numpy as np
import os
from multiprocessing import Pool
import multiprocessing as multi

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
group = "all5"
# path = "/gs/hs0/tga-science/kimura/transformer_tape_dnabert/data/"
path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/data/"
output = f"{path}training_data_tokenized/"
hbpot_path = f"{path}attn_arrays_hb/"
pipot_path = f"{path}attn_arrays_pi/"
input = f"{path}training_data/"
protein_cv_list = [['PPIL4', 'XRN2', 'PRPF4', 'SF3B1', 'NPM1', 'NIP7', 'RBM22', 'SRSF1', 'RPS5', 'EFTUD2', 'STAU2', 'KHDRBS1', 'DKC1', 'FKBP4', 'PUS1', 'YBX3', 'YWHAG', 'GTF2F1', 'KHSRP', 'DROSHA', 'LARP7', 'RPS11', 'HNRNPUL1', 'TRA2A', 'HNRNPC', 'UCHL5', 'LSM11', 'AATF', 'NOLC1', 'HNRNPK', 'ILF3'], ['NIPBL', 'RBFOX2', 'LIN28B', 'FXR1', 'TIA1', 'ZRANB2', 'APOBEC3C', 'IGF2BP1', 'BUD13', 'DHX30', 'GPKOW', 'NOL12', 'PABPN1', 'PCBP1', 'ZC3H11A', 'HLTF', 'SDAD1', 'RBM15', 'RBM27', 'RPS3', 'SRSF9', 'METAP2', 'CSTF2', 'U2AF2', 'NSUN2', 'AARS', 'EIF3D', 'TNRC6A', 'EIF3H', 'RBM5', 'DDX24'], ['HNRNPU', 'WRN', 'PPIG', 'SLTM', 'SERBP1', 'DGCR8', 'SUB1', 'EIF3G', 'EWSR1', 'SMNDC1', 'SAFB', 'TBRG4', 'DDX59', 'POLR2G', 'IGF2BP3', 'MTPAP', 'UTP3', 'BCLAF1', 'DDX3X', 'PTBP1', 'HNRNPA1', 'CPSF6', 'SAFB2', 'TROVE2', 'SUPV3L1', 'DDX21', 'DDX6', 'SUGP2', 'DDX42', 'XPO5', 'CSTF2T'], ['FUS', 'UTP18', 'SBDS', 'TAF15', 'NCBP2', 'CPEB4', 'WDR3', 'U2AF1', 'SF3B4', 'SF3A3', 'FTO', 'GRSF1', 'SLBP', 'HNRNPL', 'NKRF', 'EXOSC5', 'DDX55', 'TARDBP', 'ZC3H8', 'GRWD1', 'PUM1', 'EIF4G2', 'FMR1', 'SSB', 'XRCC6', 'G3BP1', 'AKAP8L', 'BCCIP', 'IGF2BP2', 'LARP4', 'FXR2'], ['PUM2', 'NONO', 'FASTKD2', 'PRPF8', 'PHF6', 'PABPC4', 'FAM120A', 'SFPQ', 'CDC40', 'HNRNPM', 'FUBP3', 'TIAL1', 'SRSF7', 'ZNF622', 'QKI', 'DDX52', 'GNL3', 'AGGF1', 'AKAP1', 'SND1', 'ZNF800', 'PCBP2', 'UPF1', 'MATR3', 'AUH', 'AQR', 'DDX51', 'ABCF1', 'GEMIN5', 'WDR43']]
PCODES = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
RCODES = ["A", "G", "C", "U"]

targetrange = list(range(5000))
eachlen = round(5000 / 25)
pmax = 2805

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def ptokenize(vocab_list, seq):
    unpadded = [vocab_list.index(x) + 1 if x in vocab_list else len(vocab_list) + 1 for x in seq]
    pads = [0] * (pmax - len(unpadded))
    padded = unpadded + pads
    return padded, len(unpadded)


def rtokenize(vocab_list, seq):
    tokens = [vocab_list.index(x) + 1 if x in vocab_list else len(vocab_list) + 1 for x in seq]
    return tokens


def make_pmask(protein_length):
    first = np.repeat([0], repeats=protein_length, axis=0)
    second = np.repeat([1], repeats=(pmax - protein_length), axis=0)
    self_pro_mask_row = np.concatenate([first, second])
    upper_mask = np.repeat([self_pro_mask_row], repeats=protein_length, axis=0)
    lower_mask = np.repeat([[1] * pmax], repeats=(pmax - protein_length), axis=0)
    self_pro_mask_list = np.concatenate([upper_mask, lower_mask])
    self_pro_mask_list = self_pro_mask_list[np.newaxis, :, :]
    return self_pro_mask_list


def make_cross_mask(protein_length):
    first = np.repeat([0], repeats=protein_length, axis=0)
    second = np.repeat([1], repeats=(pmax - protein_length), axis=0)
    self_cross_mask_row = np.concatenate([first, second])
    cross_mask = np.repeat([self_cross_mask_row], repeats=101, axis=0)
    cross_mask = cross_mask[np.newaxis, :, :]
    return cross_mask


def make_label_vec(label):
    if label == "1":
        new_label_list = np.array([[0, 1]], dtype="float32")
    else:
        new_label_list = np.array([[1, 0]], dtype="float32")
    return new_label_list


def main(gnum):
    if gnum == 24:
        thisrange = targetrange[eachlen * gnum:]
    else:
        thisrange = targetrange[eachlen * gnum: eachlen * (gnum + 1)]
    for i in thisrange:
        if os.path.isfile(f"{output}{group}/{i}.npz"):
            if i == 45:
                pass
            else:
                continue
        print(f"loading {i}")
        arr = np.load(f"{input}{group}/{i}.npy", allow_pickle=True)
        hbpot_arr = np.load(f"{hbpot_path}0/{i}.npz", allow_pickle=True)["pot"]
        pipot_arr = np.load(f"{pipot_path}0/{i}.npz", allow_pickle=True)["pot"]
        pro_id_arr, pro_token_arr, rna_token_arr, label_arr, pro_mask_arr, cross_mask_arr = None, None, None, None, None, None
        for idx, (item, hpot, ppot) in enumerate(zip(arr, hbpot_arr, pipot_arr)):
            print(idx)
            pro_array_tokenized, pro_len = ptokenize(PCODES, item[1])
            rna_array_tokenized = rtokenize(RCODES, item[2])
            if idx == 0:
                pro_id_arr = np.atleast_2d([int(item[0])])
                pro_token_arr = np.array(pro_array_tokenized)[np.newaxis, :]
                rna_token_arr = np.array(rna_array_tokenized)[np.newaxis, :]
                label_arr = make_label_vec(item[3])
                pro_mask_arr = make_pmask(pro_len)
                cross_mask_arr = make_cross_mask(pro_len)
                hpots = hpot[np.newaxis, :]
                ppots = ppot[np.newaxis, :]
            else:
                pro_id_arr = np.concatenate([pro_id_arr, np.atleast_2d([int(item[0])])], axis=0)
                pro_token_arr = np.concatenate([pro_token_arr, np.array(pro_array_tokenized)[np.newaxis, :]], axis=0)
                rna_token_arr = np.concatenate([rna_token_arr, np.array(rna_array_tokenized)[np.newaxis, :]], axis=0)
                label_arr = np.concatenate([label_arr, make_label_vec(item[3])], axis=0)
                pro_mask_arr = np.concatenate([pro_mask_arr, make_pmask(pro_len)], axis=0)
                cross_mask_arr = np.concatenate([cross_mask_arr, make_cross_mask(pro_len)], axis=0)
                hpots = np.concatenate([hpots, hpot[np.newaxis, :]], axis=0)
                ppots = np.concatenate([ppots, ppot[np.newaxis, :]], axis=0)

        np.savez_compressed(f"{output}{group}/{i}", proid=pro_id_arr, protok=pro_token_arr, rnatok=rna_token_arr,
                            label=label_arr, pro_mask=pro_mask_arr, cross_mask=cross_mask_arr, hb_pots=hpots, pi_pots=ppots)


def test_load():
    arr = np.load(f"{output}{group}_small/0.npz", allow_pickle=True)
    protok = arr["protok"]
    rnatok = arr["rnatok"]
    for x, y in zip(protok, rnatok):
        print(len(x))
        print(x)
        break


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':

    import sys
    p = Pool(25)
    p.map(main, range(25))
    p.close()
    # test_load()
