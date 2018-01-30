from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

number = 2
dimension = 2

all_vectors = []
all_labels = []

print("load embedding")
with open(str(number)) as openfile:
    for line in openfile:
        parts = line.strip("\r\n").split("\t")
        if parts[0] == "4":
            continue
        all_labels.append(int(parts[0]))
        all_vectors.append([float(each) for each in parts[1].split(" ")])

matrix = np.array(all_vectors, dtype="float32")

print("Do TSNE")
print(matrix.shape)
sne = TSNE(
    n_components=2,
).fit_transform(matrix)
print(sne.shape)

# plot here
colors = ["green", "red", "blue", "yellow"]

fig, ax = plt.subplots()

scale = 85
for i in range(sne.shape[0]):
    x = sne[i][0]
    y = sne[i][1]
    color = colors[all_labels[i]]

    ax.scatter(x, y, c=color, s=scale, label=color,
               alpha=1, edgecolors='none')
# ax.legend()
ax.grid(True)
plt.show()
