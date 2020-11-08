from ml_datasets.mnist import EMNIST
from ml_datasets.utils import plot_images
from numpy import unique


def main():
    dataset = "mnist"
    emnist = EMNIST(verbose=False, force=False, dataset=dataset, rotate=True)

    x_train, y_train, x_test, y_test = emnist.load()

    print(
        "x_train: {}, y_train: {}, x_test: {}, y_test: {}".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape
        )
    )

    labels = unique(y_train)[-10:]
    print(labels)
    labels = {i: i for i in labels}

    num_sample_perclass = 10
    title = "EMNIST-{} Dataset ({} samples)".format(
        dataset, num_sample_perclass * len(labels)
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
