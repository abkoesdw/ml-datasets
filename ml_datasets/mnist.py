# Copyright (c) 2020 Arief Koesdwiady
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import zipfile
import gzip
import os
import idx2numpy
from numpy import rot90
from ml_datasets.dataset import Dataset


class EMNIST(Dataset):
    """
    This is a set of handwritten character digits datasets from EMNIST.

    The details of these datasets can be found in:
        https://www.nist.gov/itl/products-and-services/emnist-dataset

    There are six different splits provided in this dataset. A short summary of the \
        dataset is provided below:

        EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
        EMNIST ByMerge: 814,255 characters. 47 unbalanced classes.
        EMNIST Balanced:  131,600 characters. 47 balanced classes.
        EMNIST Letters: 145,600 characters. 26 balanced classes.
        EMNIST Digits: 280,000 characters. 10 balanced classes.
        EMNIST MNIST: 70,000 characters. 10 balanced classes.

    Please cite the following paper when using or referencing the dataset:
        Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an \
            extension of MNIST to handwritten letters. Retrieved from \
                http://arxiv.org/abs/1702.05373

    Arguments:
        dataset: type of datasets ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
        rotate: True or False (original data are not in the right orientation)

    Returns:
        Tuple of numpy arrays: `(x_train, y_train, x_test, y_test)`
    """

    def __init__(self, *args, **kwargs):
        kwargs["url"] = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/"
        kwargs["filename"] = "gzip.zip"

        super(EMNIST, self).__init__(*args, **kwargs)

        self.check_url(self.url)
        self.dataset = kwargs.get("dataset", "mnist")
        self.rotate = kwargs.get("rotate", True)

    def load(self):
        if self.verbose:
            print("Retrieving EMNIST-{} dataset...".format(self.dataset))

        self.download_file()

        return self.__parse_file(self.target_filename)

    def __parse_file(self, filename):
        output_ = dict()
        dir_name = os.path.dirname(filename)

        with zipfile.ZipFile(filename) as f_in:
            datasets = [
                f
                for f in f_in.namelist()
                if "-" + self.dataset + "-" in f and f.endswith(".gz")
            ]

            for dataset in datasets:
                dataset_with_full_path = os.path.join(dir_name, dataset)

                if not os.path.isfile(dataset_with_full_path):
                    f_in.extract(dataset, dir_name)

                pre = [t for t in ["train", "test"] if t in dataset_with_full_path][0]
                post = [t for t in ["images", "labels"] if t in dataset_with_full_path][
                    0
                ]
                name = pre + "_" + post

                with gzip.open(dataset_with_full_path, "rb") as f:
                    array_temp = idx2numpy.convert_from_string(f.read())
                    if post == "images":
                        if self.rotate:
                            array_temp = rot90(array_temp, k=-1, axes=(-2, -1))[
                                ..., ::-1
                            ]

                    output_[name] = array_temp

        return (
            output_["train_images"],
            output_["train_labels"],
            output_["test_images"],
            output_["test_labels"],
        )


