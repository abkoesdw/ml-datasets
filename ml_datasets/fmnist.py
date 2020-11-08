import os
import tempfile
import idx2numpy
import gzip
from ml_datasets.dataset import Dataset
from numpy import rot90


class FashionMNIST(Dataset):
    def __init__(self, *args, **kwargs):
        kwargs["url"] = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        kwargs["filename"] = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]
        kwargs["target_dir"] = kwargs.get(
            "target_dir", os.path.join(tempfile.gettempdir(), "fmnist")
        )

        super(FashionMNIST, self).__init__(*args, **kwargs)

        self.check_url(self.url)
        self.dataset = kwargs.get("dataset", "mnist")
        self.rotate = kwargs.get("rotate", False)
        self.meta = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot",
        }

    def load(self):
        if self.verbose:
            print("Retrieving Fashion-MNIST-{} dataset...".format(self.dataset))

        self.download_file()

        return self.__parse_file(self.target_filename)

    def __parse_file(self, filenames):
        output_ = dict()

        for filename in filenames:
            pre = "train" if "train" in filename else "test"
            post = [t for t in ["images", "labels"] if t in filename][0]
            name = pre + "_" + post

            with gzip.open(filename, "rb") as f:
                array_temp = idx2numpy.convert_from_string(f.read())

                if post == "images":

                    if self.rotate:
                        array_temp = rot90(array_temp, k=-1, axes=(-2, -1))[..., ::-1]

                output_[name] = array_temp

        return (
            output_["train_images"],
            output_["train_labels"],
            output_["test_images"],
            output_["test_labels"],
        )
