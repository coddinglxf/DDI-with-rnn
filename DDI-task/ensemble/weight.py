import torch

openfile = open("cast.txt")
datasets = []
length = 0
for line in openfile:
    parts = line.strip("\r\n").split("\t")
    label = int(parts[0])
    weights = []
    length = len(parts) - 1
    for vector in parts[1:]:
        vectors = [float(vec) for vec in vector.split(" ")]
        weights.append(vectors)
    datasets.append((weights, label))
    print(length)

#
