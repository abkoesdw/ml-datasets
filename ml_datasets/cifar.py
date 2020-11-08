import os
import tarfile
import pickle
from numpy import array, vstack
from ml_datasets.dataset import Dataset


class CIFAR10(Dataset):
    def __init__(self, *args, **kwargs):
        kwargs["url"] = "https://www.cs.toronto.edu/~kriz/"
        kwargs["filename"] = "cifar-10-python.tar.gz"

        super(CIFAR10, self).__init__(*args, **kwargs)

        self.check_url(self.url)

    def load(self):
        if self.verbose:
            print("Retrieving CIFAR-10 dataset...")
        self.download_file()
        return self.__parse_file(self.target_filename)

    def __parse_file(self, filename):
        dir_name = os.path.dirname(filename)
        x_train = array([])
        y_train = array([])
        x_test = array([])
        y_test = array([])

        with tarfile.open(filename) as f_in:
            datasets = [f for f in f_in.getnames() if "_batch" in f]

            for dataset in datasets:
                dataset_full_path = os.path.join(dir_name, dataset)

                if not os.path.isfile(dataset_full_path):
                    f_in.extract(dataset, path=dir_name)

                with open(dataset_full_path, "rb") as f:
                    temp = pickle.load(f, encoding="bytes")
                    temp = self.bytes_to_utf(temp)
                    x_temp = temp["data"]
                    y_temp = array(temp["labels"]).reshape(-1, 1)

                    if "test" not in dataset:
                        x_train = vstack([x_train, x_temp]) if x_train.size else x_temp
                        y_train = vstack([y_train, y_temp]) if y_train.size else y_temp

                    else:
                        x_test = x_temp
                        y_test = y_temp

            meta = [f for f in f_in.getnames() if "meta" in f][0]
            meta_full_path = os.path.join(dir_name, meta)

            if not os.path.isfile(meta_full_path):
                f_in.extract(meta, path=dir_name)

            with open(meta_full_path, "rb") as f:
                self.meta = pickle.load(f, encoding="bytes")
                self.meta = self.bytes_to_utf(self.meta)
                self.meta["label_names"] = [
                    x.decode("utf-8") for x in self.meta["label_names"]
                ]

        return (
            x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
            y_train.reshape(-1),
            x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
            y_test.reshape(-1),
        )

    def bytes_to_utf(self, data):
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, dict):
            return dict(map(self.bytes_to_utf, data.items()))
        if isinstance(data, tuple):
            return map(self.bytes_to_utf, data)
        return data


class CIFAR100(Dataset):
    def __init__(self, *args, **kwargs):
        kwargs["url"] = "https://www.cs.toronto.edu/~kriz/"
        kwargs["filename"] = "cifar-100-python.tar.gz"
        super(CIFAR100, self).__init__(*args, **kwargs)

        self.check_url(self.url)
        self.labels_type = kwargs.get("labels_type", "fine")

    def load(self):
        self.download_file()
        return self.__parse_file(self.target_filename)

    def __parse_file(self, filename):
        dir_name = os.path.dirname(filename)
        x_train = array([])
        y_train = array([])
        x_test = array([])
        y_test = array([])

        with tarfile.open(filename) as f_in:
            datasets = [f for f in f_in.getnames() if "train" in f or "test" in f]

            for dataset in datasets:
                dataset_full_path = os.path.join(dir_name, dataset)

                if not os.path.isfile(dataset_full_path):
                    f_in.extract(dataset, path=dir_name)

                with open(dataset_full_path, "rb") as f:
                    temp = pickle.load(f, encoding="bytes")
                    temp = self.bytes_to_utf(temp)
                    x_temp = temp["data"]
                    y_temp = array(temp[self.labels_type + "_labels"]).reshape(-1, 1)

                    if "test" not in dataset:
                        x_train = vstack([x_train, x_temp]) if x_train.size else x_temp
                        y_train = vstack([y_train, y_temp]) if y_train.size else y_temp

                    else:
                        x_test = x_temp
                        y_test = y_temp

            meta = [f for f in f_in.getnames() if "meta" in f][0]
            meta_full_path = os.path.join(dir_name, meta)

            if not os.path.isfile(meta_full_path):
                f_in.extract(meta, path=dir_name)

            with open(meta_full_path, "rb") as f:
                self.meta = pickle.load(f, encoding="bytes")
                self.meta = self.bytes_to_utf(self.meta)
                self.meta["coarse_label_names"] = [
                    x.decode("utf-8") for x in self.meta["coarse_label_names"]
                ]
                self.meta["fine_label_names"] = [
                    x.decode("utf-8") for x in self.meta["fine_label_names"]
                ]

        # if self.labels_type == "coarse":

        return (
            x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
            y_train.reshape(-1),
            x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
            y_test.reshape(-1),
        )

    def bytes_to_utf(self, data):
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, dict):
            return dict(map(self.bytes_to_utf, data.items()))
        if isinstance(data, tuple):
            return map(self.bytes_to_utf, data)
        return data
