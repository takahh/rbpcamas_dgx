path = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/data/154proteins_w_uniprotID.fasta"
opath = "/Users/mac/Desktop/t3_mnt/transformer_tape_dnabert/AttentionAnalysis/data/154proteins_w_uniprotID_length.txt"
lendict = {}


def main():
  name, sequence = "", ""
  with open(path) as f, open(opath, "w") as fo:
    for lines in f.readlines():
      if ">" in lines:
        if name is not None:
          lendict[name] = len(sequence)
          # fo.writelines(f"{name}:{len(sequence)}\n")
        name = lines[1:].split()[0]
        sequence = None
      else:
        if sequence is not None:
          sequence += lines.strip()
        else:
          sequence = lines.strip()

    newdict = sorted(lendict.items(), key=lambda x : x[1])
    print(newdict)
    for item in newdict[1:]:
      fo.writelines(f"{item[0]}: {item[1]}\n")


if __name__ == "__main__":
  main()