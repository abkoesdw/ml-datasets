from ml_datasets.cifar import CIFAR10
from ml_datasets.utils import plot_images


def main():
    cifar10 = CIFAR10(verbose=False, force=False)
    x_train, y_train, x_test, y_test = cifar10.load()

    print(
        "x_train: {}, y_train: {}, x_test: {}, y_test: {}".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape
        )
    )

    print(cifar10.meta)

    num_sample_perclass = 20
    title = "CIFAR-10 Dataset ({} samples)".format(
        num_sample_perclass * len(cifar10.meta["label_names"])
    )
    labels = {i: j for i, j in enumerate(cifar10.meta["label_names"])}

    plot_images(
        num_sample_perclass=num_sample_perclass,
        x=x_train,
        y=y_train,
        labels=labels,
        title=title,
    )


if __name__ == "__main__":
    main()
