import matplotlib.pyplot as plt


def plot_attention(data, x, y):
    fig, ax = plt.subplots(figsize=(10, 1))
    heatmp = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)

    xticks = range(0, len(x))

    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels(x, minor=False, fontdict={"size": 10})

    yticks = range(0, len(y))
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(y, minor=False, )

    ax.grid(True)
    plt.show()


import numpy as np

x = "Synergism was also noted when DRUG1 was combined with DRUG2 and DRUG0 ."
y = "value"

# data = np.random.rand(len(y.split(" ")), len(x.split(" ")))

data = "0.158301 0.0969846 0.0832293 0.0821719 0.0849167 0.0756122 0.0707429 0.089885 0.0902637 0.0392366 0.0115379 0.0252844 0.0918337".split(
    " ")
data = [float(each) for each in data]
data = np.array(data).reshape((1, -1))

plot_attention(data, x.split(" "), y.split(" "))
