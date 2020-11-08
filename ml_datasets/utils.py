import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm


def plot_images(
    num_sample_perclass=10, x=None, y=None, labels=None, title=None, cmap=None
):
    grid_x = num_sample_perclass + 1
    grid_y = len(labels)

    plt.figure(figsize=(grid_y, grid_x))
    gs1 = gridspec.GridSpec(grid_y, grid_x)
    gs1.update(wspace=0.025, hspace=0.05)

    font = {"family": "serif", "weight": "bold"}

    plt.suptitle(title)
    j = 0
    for i in range(grid_y):
        idxs = [0] + list(np.where(y == list(labels.keys())[i])[0][: grid_x - 1])
        label = labels[list(labels.keys())[i]]

        for k, idx in enumerate(idxs):
            ax1 = plt.subplot(gs1[j])

            if k == 0:
                ax1.text(0, 0.25, label, ha="right", wrap=True, fontdict=font)

            else:
                ax1.imshow(x[idx, ...], cmap=cmap)

            plt.axis("off")
            j += 1

    plt.show()


def plot_2D(x, y, title, axis="off"):
    BLUE, ORANGE = "#57B5E8", "#E69E00"
    plt.figure(figsize=(8, 8))
    plt.scatter(
        x[:, 0],
        x[:, 1],
        s=18,
        facecolors="none",
        edgecolors=np.array([BLUE, ORANGE])[y],
    )
    if axis == "off":
        plt.axis("off")
    elif axis == "on":
        plt.xlabel("x_1")
        plt.ylabel("x_2")
    else:
        print("incorrect values for arg: axis (on or off only)")
        sys.exit()

    plt.title(title)
    plt.show()


def plot_dna(df, label):
    matrix = df.values
    col_names = df.columns
    rows = np.arange(matrix.shape[0])
    cols = np.arange(matrix.shape[1])
    np.random.seed(3)
    np.random.shuffle(rows)
    np.random.shuffle(cols)

    matrix = matrix[:, cols[:100]].T
    matrix = matrix[:, rows]
    col_names = col_names[cols[:100]]
    label = label[rows]
    mat_min = np.min(matrix)
    mat_max = np.max(matrix)
    mat_min = -np.max([np.abs(mat_min), mat_max])
    mat_max = np.max([np.abs(mat_min), mat_max])
    matrix = np.ma.masked_where(np.abs(matrix) <= 0.3, matrix)

    plt.figure(figsize=(6, 12))
    cmap_list = ["red", "darkred", "green", "lime", "lightgreen"]
    cmap = LinearSegmentedColormap.from_list("Custom cmap", cmap_list, len(cmap_list))
    cmap.set_bad("black")

    bounds = np.linspace(
        mat_min + 6, mat_max - 6, 5
    )  # np.arange(mat_min + 6, mat_max - 6, 0.1)
    idx = np.searchsorted(bounds, 0)

    bounds = np.insert(bounds, idx, 0)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(matrix, cmap=cmap, norm=norm)
    plt.xticks(np.arange(len(label)))
    plt.yticks(np.arange(len(col_names)))
    ax = plt.gca()
    ax.set_xticklabels(label, rotation=90)
    ax.set_yticklabels(col_names)
    ax.yaxis.tick_right()
    ax.tick_params(axis=u"both", which=u"both", labelsize=5, length=0.0)
    plt.tight_layout()
    fig = plt.gcf()
    # fig.set_size_inches((6, 12), forward=False)
    # fig.savefig("img/dna.png", dpi=200)
    plt.show()
