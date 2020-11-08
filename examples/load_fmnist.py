from ml_datasets.fmnist import FashionMNIST
from ml_datasets.utils import plot_images


def main():
    fmnist = FashionMNIST(verbose=False, force=False)
    x_train, y_train, x_test, y_test = fmnist.load()

    print(
        "x_train: {}, y_train: {}, x_test: {}, y_test: {}".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape
        )
    )

    labels = fmnist.meta

    num_sample_perclass = 20
    title = "Fashion-MNIST Dataset ({} samples)".format(
        num_sample_perclass * len(labels)
    )

    plot_images(
        num_sample_perclass=num_sample_perclass,
        x=x_train,
        y=y_train,
        labels=labels,
        title=title,
        cmap="Greys_r",
    )


if __name__ == "__main__":
    main()
