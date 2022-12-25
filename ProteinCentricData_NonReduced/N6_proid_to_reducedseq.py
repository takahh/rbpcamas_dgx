# -------------------------------------------------------------------
# this code writes token list of [RNA and protein and label]
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
import os, random
import numpy as np

# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------


one2three_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

reduce_resi_dict = {
    20: ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
          'SER', 'THR', 'TRP', 'TYR', 'VAL'],
    13: ['TYR', 'LEU', 'LYS', 'GLY', 'TRP', 'ILE', 'ARG', 'PHE', 'CYS', 'HIS', 'ASP', 'ALA', 'ASN'],
    8: ['TYR', 'GLY', 'ILE', 'ARG', 'ASN', 'PHE', 'HIS', 'LYS'],
    4: ['LYS', 'ARG', 'PHE', 'ILE']
}

group = 0

three2one_dict = {}
for item in one2three_dict.keys():
    three2one_dict[one2three_dict[item]] = item

np.random.seed(0)

# ------------------------  CHECK AND CHANGE HERE IF NECESSARY !!!!!!
mode = "unknown_protein_broad_share"  # all/lnc/all_unknown_protein/shared_RNA
REDUCE_LEVEL = 8  # 20, 13, 8, 4
# ---------------------------------------
inputpath = "/Users/mac/Documents/RBP_CAMAS/data/newdata/freq25_32_with_negatives_all101.csv"
output = f"/Users/mac/Documents/RBP_CAMAS/data/newdata/batched_nparray/nored/"

count_per_file = 5

protein_cv_list_2805 = [['EIF3D', 'HNRNPL', 'BCLAF1', 'CPEB4', 'NSUN2', 'SMNDC1', 'AKAP8L', 'BCCIP', 'EIF3G', 'DDX24', 'METAP2', 'EIF4G2', 'LSM11', 'UCHL5', 'UTP18', 'AKAP1', 'SND1', 'DDX42', 'IGF2BP3', 'PTBP1', 'RBFOX2', 'GPKOW', 'SRSF7', 'STAU2', 'TAF15', 'SAFB2', 'ZC3H8', 'SF3B4', 'HNRNPA1', 'XRN2', 'SUGP2'], ['XPO5', 'LIN28B', 'NONO', 'SDAD1', 'ZNF622', 'SFPQ', 'SF3B1', 'UTP3', 'CPSF6', 'LARP7', 'NIPBL', 'ZRANB2', 'KHSRP', 'WRN', 'HNRNPC', 'SRSF1', 'EIF3H', 'ILF3', 'PPIG', 'CSTF2T', 'LARP4', 'SSB', 'AGGF1', 'U2AF2', 'SF3A3', 'YWHAG', 'FMR1', 'U2AF1', 'NCBP2', 'DKC1', 'TRA2A'], ['SERBP1', 'ZC3H11A', 'FXR2', 'FXR1', 'SUPV3L1', 'FKBP4', 'RPS11', 'GRSF1', 'ZNF800', 'SUB1', 'PPIL4', 'FUBP3', 'WDR3', 'NOLC1', 'UPF1', 'HNRNPU', 'HNRNPUL1', 'DROSHA', 'DDX52', 'QKI', 'AUH', 'TROVE2', 'DDX51', 'ABCF1', 'TBRG4', 'DDX21', 'XRCC6', 'GTF2F1', 'DGCR8', 'CDC40', 'MATR3'], ['PHF6', 'POLR2G', 'NIP7', 'WDR43', 'DHX30', 'FUS', 'MTPAP', 'NPM1', 'HNRNPK', 'SRSF9', 'PABPC4', 'PRPF4', 'DDX3X', 'RPS3', 'AATF', 'SBDS', 'DDX59', 'RBM15', 'DDX6', 'PUM2', 'EXOSC5', 'HLTF', 'RBM27', 'PUM1', 'PUS1', 'GEMIN5', 'GRWD1', 'RBM5', 'CSTF2', 'FTO', 'SLTM'], ['GNL3', 'EWSR1', 'AARS', 'NKRF', 'FAM120A', 'PCBP2', 'FASTKD2', 'EFTUD2', 'PRPF8', 'RBM22', 'IGF2BP2', 'BUD13', 'SAFB', 'TIAL1', 'NOL12', 'IGF2BP1', 'PCBP1', 'SLBP', 'AQR', 'DDX55', 'APOBEC3C', 'G3BP1', 'KHDRBS1', 'PABPN1', 'TARDBP', 'HNRNPM', 'TIA1', 'YBX3', 'RPS5', 'TNRC6A']]
# unknown_cv_list = [['SERBP1', 'KHDRBS1', 'SAFB2', 'SFPQ', 'CSTF2', 'ZC3H8', 'NSUN2', 'GTF2F1', 'ABCF1', 'POLR2G', 'LARP4', 'CDC40', 'DDX59', 'LSM11', 'PHF6', 'DDX6', 'TIAL1', 'IGF2BP1', 'DDX55', 'DDX42', 'GEMIN5', 'AUH', 'METAP2', 'SAFB', 'DKC1', 'SUB1', 'STAU2', 'SSB', 'FXR2', 'PCBP1', 'UTP18'], ['HNRNPU', 'ILF3', 'MATR3', 'CSTF2T', 'U2AF2', 'RPS11', 'NIP7', 'EIF3D', 'AGGF1', 'XPO5', 'DGCR8', 'FTO', 'FASTKD2', 'BUD13', 'RBM15', 'SF3B1', 'PUM2', 'SUPV3L1', 'PUS1', 'FXR1', 'DDX3X', 'EIF4G2', 'SRSF9', 'EIF3H', 'DDX21', 'FUS', 'RPS3', 'PABPC4', 'DDX24', 'DDX52', 'SDAD1'], ['HNRNPA1', 'KHSRP', 'HNRNPUL1', 'HNRNPM', 'TROVE2', 'XRN2', 'PCBP2', 'SND1', 'RBM5', 'CPEB4', 'U2AF1', 'XRCC6', 'UCHL5', 'HLTF', 'AKAP8L', 'RBM22', 'DHX30', 'TARDBP', 'FKBP4', 'GRWD1', 'ZNF622', 'IGF2BP3', 'TNRC6A', 'ZRANB2', 'IGF2BP2', 'PPIG', 'NOL12', 'BCLAF1', 'PUM1', 'WDR43', 'ZNF800'], ['HNRNPC', 'WDR3', 'TIA1', 'NONO', 'HNRNPK', 'AATF', 'NKRF', 'SLTM', 'PPIL4', 'DROSHA', 'TBRG4', 'NPM1', 'BCCIP', 'NCBP2', 'DDX51', 'PRPF8', 'UPF1', 'LARP7', 'SMNDC1', 'YWHAG', 'SRSF7', 'LIN28B', 'CPSF6', 'YBX3', 'TAF15', 'TRA2A', 'AQR', 'AKAP1', 'MTPAP', 'NIPBL', 'PRPF4'], ['SLBP', 'SBDS', 'SUGP2', 'QKI', 'RBFOX2', 'FUBP3', 'EXOSC5', 'EWSR1', 'FAM120A', 'EIF3G', 'PTBP1', 'SRSF1', 'NOLC1', 'WRN', 'EFTUD2', 'RPS5', 'UTP3', 'GPKOW', 'AARS', 'RBM27', 'GRSF1', 'SF3B4', 'HNRNPL', 'GNL3', 'ZC3H11A', 'SF3A3', 'PABPN1', 'FMR1', 'G3BP1', 'APOBEC3C']]
proseq_master = f"/Users/mac/Documents/transformer_tape_dnabert/data/proteinseqs.csv"

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def get_pro_id_dict():
    id = 0
    id_dict = {}
    for item in protein_cv_list_2805:
        for name in item:
            id_dict[name] = id
            id += 1
    return id_dict


def get_proseq(proname):
    sequence = None
    with open(proseq_master) as f:
        for lines in f.readlines():
            if proname == lines.split(",")[0]:
                sequence = lines.split(",")[1].strip()
    return sequence


def reset_dirs(output):
    if mode == "unknown_protein_broad_share":
        for files in os.listdir(output):
            if os.path.isdir(f"{output}/{files}"):
                continue
            try:
                os.remove(f"{output}/{files}")
            except FileNotFoundError:
                pass
    if not os.path.exists(output):
        try:
            os.mkdir(output)
        except FileExistsError:
            pass


def runs6(max_posi_count_per_rna=10):
    reset_dirs(output)
    datalist = []
    pro_id_dict = get_pro_id_dict()
    count = 0
    rna_max_count = 5000//(max_posi_count_per_rna * 2) + 2
    with open(inputpath) as f:
        # shuffle RNAs
        flines = f.readlines()
        random.shuffle(flines)
        for idx, lines in enumerate(flines):
            if idx > rna_max_count:
                break
            # UUUAU...GAGACAU,GTF2F1.csvchr3-129387171-12938727222"['RBFOX2' 'FUS' ... 'DDX6']"{'POLR2G'...}
            ele = lines.split(",")
            rnaseq = ele[0]
            posi_proteins = ele[1].split("[")[1].split("]")[0].replace("'", "").split()
            # ele[1] : HNRNPU.csvchr3-14960-14121"['THNRNPA1''SFPQ']"{'DDX3X' 'U2AF2'}
            nega_proteins = ele[1].split("[")[1].split("]")[0].replace("'", "").split()
            # print(f"{len(posi_proteins)} + {len(nega_proteins)} = {len(posi_proteins) + len(nega_proteins)}")
            # common_least_count = rna_max_count
            # # take at least 40 posi and 40 nega
            # if common_least_count < 80:
            #     continue
            # random.shuffle(nega_proteins)
            # random.shuffle(posi_proteins)
            # selected_negas = nega_proteins[:common_least_count]
            # selected_posis = posi_proteins[:common_least_count]
            selected_negas = nega_proteins
            selected_posis = posi_proteins
            # ------------------------------------
            # make data from positive proteins
            # ------------------------------------
            for pronames in selected_posis:
                proid = pro_id_dict[pronames]
                proseq = get_proseq(pronames)
                datalist.append([proid, proseq, rnaseq, 1])
            # ------------------------------------
            # make data from negative proteins
            # ------------------------------------
            for pronames in selected_negas:
                try:
                    proid = pro_id_dict[pronames]
                except KeyError:
                    continue
                proseq = get_proseq(pronames)
                datalist.append([proid, proseq, rnaseq, 0])
        # -----------
        # batchfy
        # -----------
        final_arr = np.array(datalist)
        itermax = final_arr.shape[0] // 5

        for i in range(itermax):
            # if final_arr[i * count_per_file: (i + 1) * count_per_file].shape[0] != 5:
            #     print(f"final arr {i} - {final_arr[i * count_per_file: (i + 1) * count_per_file].shape}")
            np.save(f"{output}/{i}", final_arr[i * count_per_file: (i + 1) * count_per_file])


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    runs6()
