from ml_datasets.cifar import CIFAR100
from ml_datasets.utils import plot_images


def main():
    labels_type = "coarse"
    cifar100 = CIFAR100(verbose=False, force=False, labels_type=labels_type)
    x_train, y_train, x_test, y_test = cifar100.load()

    print(
        "x_train: {}, y_train: {}, x_test: {}, y_test: {}".format(
            x_train.shape, y_train.shape, x_test.shape, y_test.shape
        )
    )

    num_sample_perclass = 10
    title = "CIFAR-100 Dataset {} labels({} samples)".format(
        labels_type,
        num_sample_perclass * len(cifar100.meta[labels_type + "_label_names"]),
    )
    labels = {
        i: j for i, j in enumerate(cifar100.meta[labels_type + "_label_names"][:20])
    }

    plot_images(
        num_sample_perclass=num_sample_perclass,
        x=x_train,
        y=y_train,
        labels=labels,
        title=title,
    )


if __name__ == "__main__":
    main()
