from ml_datasets.esl import Mixture
from ml_datasets.esl import ProstateCancer
from ml_datasets.esl import EmailSpam
from ml_datasets.esl import HandwrittenDigit
from ml_datasets.esl import NCI
from ml_datasets.utils import plot_2D, plot_dna


def main(**kwargs):
    dataset = kwargs.get("dataset", None)

    if dataset == "mixture":
        mixture = Mixture(
            verbose=False,
            force=False,
        )
        x, y = mixture.load()

        print("x: {}, y: {}".format(x.shape, y.shape))

        plot_2D(x, y, "ESL-Mixture Dataset")

    elif dataset == "prostate":
        prostate_cancer = ProstateCancer(
            verbose=False,
            force=False,
        )
        df = prostate_cancer.load()

        print("columns: {}".format(prostate_cancer.meta))
        print(prostate_cancer.info)

    elif dataset == "spam":
        spam = EmailSpam(
            verbose=False,
            force=False,
        )
        df, x_train, y_train, x_test, y_test = spam.load()

        print("x_train: {}, y_train: {}".format(x_train.shape, y_train.shape))
        print("x_test: {}, y_test: {}".format(x_test.shape, y_test.shape))

    elif dataset == "digit":
        digit = HandwrittenDigit(
            verbose=False,
            force=False,
        )
        x_train, y_train, x_test, y_test = digit.load()

        print("x_train: {}, y_train: {}".format(x_train.shape, y_train.shape))
        print("x_test: {}, y_test: {}".format(x_test.shape, y_test.shape))

    elif dataset == "nci":
        nci = NCI(
            verbose=False,
            force=False,
        )
        df, label = nci.load()

        plot_dna(df, label)


if __name__ == "__main__":
    available_dataset = ["mixture", "prostate", "spam", "digit", "nci"]
    dataset = available_dataset[4]
    main(dataset=dataset)
