import os
import tempfile
import warnings
import pandas as pd
import numpy as np
import rdata
import gzip
from ml_datasets.dataset import Dataset

warnings.filterwarnings("ignore")


class Mixture(Dataset):
    def __init__(self, *args, **kwargs):
        kwargs["url"] = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/"
        kwargs["filename"] = "ESL.mixture.rda"
        kwargs["means"] = kwargs.get("means", 0)
        kwargs["target_dir"] = kwargs.get(
            "target_dir", os.path.join(tempfile.gettempdir(), "ESL")
        )

        super(Mixture, self).__init__(*args, **kwargs)

        self.check_url(self.url)
        self.means = kwargs["means"]

    def load(self):
        if self.verbose:
            print("Retrieving ESL-Mixture dataset...")

        self.download_file()

        return self.__parse_file(self.target_filename)

    def __parse_file(self, target_filename):
        parsed = rdata.parser.parse_file(open(target_filename))
        converted = rdata.conversion.convert(parsed)
        if self.means == 1:
            return (
                converted["ESL.mixture"]["x"],
                converted["ESL.mixture"]["y"].astype(int),
                converted["ESL.mixture"]["means"],
            )

        else:
            return (
                converted["ESL.mixture"]["x"],
                converted["ESL.mixture"]["y"].astype(int),
            )


class ProstateCancer(Dataset):
    def __init__(self, *args, **kwargs):
        kwargs["url"] = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/"
        kwargs["filename"] = ["prostate.data", "prostate.info.txt"]
        kwargs["means"] = kwargs.get("means", 0)
        kwargs["target_dir"] = kwargs.get(
            "target_dir", os.path.join(tempfile.gettempdir(), "ESL")
        )

        super(ProstateCancer, self).__init__(*args, **kwargs)

        self.check_url(self.url)

    def load(self):
        if self.verbose:
            print("Retrieving ESL-Prostate_Cancer dataset...")

        self.download_file()

        return self.__parse_file(self.target_filename)

    def __parse_file(self, target_filename):
        df = pd.read_csv(target_filename[0], sep="\t").drop("Unnamed: 0", axis=1)

        self.train_test = df["train"].values
        df.drop("train", axis=1, inplace=True)
        self.meta = list(df.columns)

        with open(target_filename[1], "r") as f:
            self.info = f.read()

        return df


class EmailSpam(Dataset):
    def __init__(self, *args, **kwargs):
        kwargs["url"] = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/"
        kwargs["filename"] = ["spam.data", "spam.info.txt", "spam.traintest"]
        kwargs["means"] = kwargs.get("means", 0)
        kwargs["target_dir"] = kwargs.get(
            "target_dir", os.path.join(tempfile.gettempdir(), "ESL")
        )

        super(EmailSpam, self).__init__(*args, **kwargs)

        self.check_url(self.url)

    def load(self):
        if self.verbose:
            print("Retrieving ESL-Email_Spam dataset...")

        self.download_file()

        return self.__parse_file(self.target_filename)

    def __parse_file(self, target_filename):
        data = pd.read_csv(target_filename[0], sep=" ", header=None)
        with open(target_filename[1], "r") as f:
            self.info = f.read()

        columns = [
            "make",
            "address",
            "all",
            "3d",
            "our",
            "over",
            "remove",
            "internet",
            "order",
            "mail",
            "receive",
            "will",
            "people",
            "report",
            "addresses",
            "free",
            "business",
            "email",
            "you",
            "credit",
            "your",
            "font",
            "000",
            "money",
            "hp",
            "hpl",
            "george",
            "650",
            "lab",
            "labs",
            "telnet",
            "857",
            "data",
            "415",
            "85",
            "technology",
            "1999",
            "parts",
            "pm",
            "direct",
            "cs",
            "meeting",
            "original",
            "project",
            "re",
            "edu",
            "table:",
            "conference",
            ";",
            "(",
            "[",
            "!",
            "$",
            "#",
        ] + [
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
            "spam",
        ]
        data.columns = columns

        self.train_test = pd.read_csv(target_filename[2], sep=" ", header=None).values
        idx_train = np.where(self.train_test == 0)[0]
        idx_test = np.where(self.train_test == 1)[0]

        x_train = data[columns[:-1]].values[idx_train]
        y_train = data[columns[-1]].values[idx_train]

        x_test = data[columns[:-1]].values[idx_test]
        y_test = data[columns[-1]].values[idx_test]

        return data, x_train, y_train, x_test, y_test


class HandwrittenDigit(Dataset):
    def __init__(self, *args, **kwargs):
        kwargs["url"] = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/"
        kwargs["filename"] = [
            "zip.info.txt",
            "zip.train.gz",
            "zip.test.gz",
            "zip.digits",
        ]
        kwargs["means"] = kwargs.get("means", 0)
        kwargs["target_dir"] = kwargs.get(
            "target_dir", os.path.join(tempfile.gettempdir(), "ESL")
        )

        super(HandwrittenDigit, self).__init__(*args, **kwargs)

        self.check_url(self.url)

    def load(self):
        if self.verbose:
            print("Retrieving ESL-Handwritten_Digit dataset...")

        self.download_file()

        return self.__parse_file(self.target_filename)

    def __parse_file(self, target_filename):
        x_train = []
        y_train = []
        with gzip.GzipFile(target_filename[1], "rb") as f:
            for line in f.readlines():
                temp = [
                    float(i) for i in line.decode("utf-8").rstrip("\n").split(" ")[:-1]
                ]
                temp_y = temp[0]
                temp_x = temp[1:]
                x_train.append(temp_x)
                y_train.append(temp_y)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_test = []
        y_test = []
        with gzip.GzipFile(target_filename[2], "rb") as f:
            for line in f.readlines():
                temp = [float(i) for i in line.decode("utf-8").rstrip("\n").split(" ")]
                temp_y = temp[0]
                temp_x = temp[1:]
                x_test.append(temp_x)
                y_test.append(temp_y)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        with open(target_filename[0], "r") as f:
            self.info = f.read()

        return x_train, y_train, x_test, y_test


class NCI(Dataset):
    def __init__(self, *args, **kwargs):
        kwargs["url"] = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/"
        kwargs["filename"] = ["nci.data.csv", "nci.label.txt", "nci.info.txt"]
        kwargs["means"] = kwargs.get("means", 0)
        kwargs["target_dir"] = kwargs.get(
            "target_dir", os.path.join(tempfile.gettempdir(), "ESL")
        )

        super(NCI, self).__init__(*args, **kwargs)

        self.check_url(self.url)

    def load(self):
        if self.verbose:
            print("Retrieving ESL-NCI dataset...")

        self.download_file()

        return self.__parse_file(self.target_filename)

    def __parse_file(self, target_filename):
        df = pd.read_csv(target_filename[0])
        df.set_index("Unnamed: 0", drop=True, inplace=True)
        df = df.transpose()

        with open(target_filename[1], "r") as f:
            label = np.loadtxt(f, dtype=np.str)

        with open(target_filename[2], "r") as f:
            self.info = f.read()

        return df, label
