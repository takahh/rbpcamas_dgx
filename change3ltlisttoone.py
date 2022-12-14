# -------------------------------------------------------------------
# this code 
# input : 
# output: 
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# import
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# constant
# -------------------------------------------------------------------
one2three_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
three2one_dict = {}
for keys in one2three_dict.keys():
    three2one_dict[one2three_dict[keys]] = keys
print(three2one_dict)


# a = ['ASP', 'THR', 'GLU', 'ILE', 'PRO', 'TYR', 'SER', 'TRP', 'GLN', 'CYS', 'MET', 'ALA', 'HIS']
# b = ['ASP', 'GLU', 'PRO', 'TYR', 'TRP', 'GLN', 'CYS', 'MET', 'HIS']
# c = ['ASP', 'GLU', 'PRO', 'TYR', 'TRP', 'HIS']

a = ['TYR', 'LEU', 'LYS', 'GLY', 'TRP', 'ILE', 'ARG', 'PHE', 'CYS', 'HIS', 'ASP', 'ALA', 'ASN']
b = ['TYR', 'GLY', 'ILE', 'ARG', 'ASN', 'PHE', 'HIS', 'LYS']
c = ['LYS', 'ARG', 'PHE', 'ILE']

# -------------------------------------------------------------------
# function
# -------------------------------------------------------------------


def main():
    newa = [three2one_dict[item] for item in a]
    newb = [three2one_dict[item] for item in b]
    newc = [three2one_dict[item] for item in c]
    print(newa)
    print(newb)
    print(newc)



# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
if __name__ == '__main__':
    main()