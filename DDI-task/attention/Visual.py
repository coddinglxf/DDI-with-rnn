import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file',
                    type=str,
                    default="/lustre2/sli/leihua/2017-new-start/PyTorch/External/"
                            "DDI_ensemble_new_experiments/attention/64_9_0.7155622776241654")
args = parser.parse_args()

with open(args.file) as openfile:
    for line in openfile:

        parts = line.strip("\r\n").split("\t")
        assert len(parts) == 4
        p = int(parts[0])
        t = int(parts[1])
        att = [float(vec) for vec in parts[2].split(" ")]
        words = [word for word in parts[3].split(" ")]

        if t != 4 and t != p:
            # print(parts[3])
            argIndex = np.argsort(att)[::-1]
            print(t, p)
            print(" ".join(words))
            # for index in argIndex:
            #     print(index, att[index], words[index], len(att), len(words))
            print("------------------------------------------------------------------------\n")
