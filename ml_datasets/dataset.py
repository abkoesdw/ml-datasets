import tempfile
import requests
import sys
import os
from tqdm import tqdm


class Dataset:
    def __init__(self, *args, **kwargs):
        self.filename = kwargs.get("filename", None)
        self.url = kwargs.get("url", None)
        self.target_dir = kwargs.get("target_dir", None)
        self.target_dir = self.target_dir or tempfile.gettempdir()

        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)

        if isinstance(self.filename, list):
            self.target_filename = []
            for i, file_ in enumerate(self.filename):
                self.target_filename.append(os.path.join(self.target_dir, file_))

        else:
            self.target_filename = os.path.join(self.target_dir, self.filename)

        self.force = kwargs.get("force", False)
        self.chunk_size = kwargs.get("chunk_size", 1024)
        self.verbose = kwargs.get("verbose", True)

    def download_file(self):
        if isinstance(self.target_filename, list):
            for i in range(len(self.target_filename)):
                self._download_file(self.filename[i], self.target_filename[i])

        else:
            self._download_file(self.filename, self.target_filename)

    def _download_file(self, filename, target_filename):
        if self.force or not os.path.isfile(target_filename):
            url = requests.compat.urljoin(self.url, filename)

            if self.verbose:
                print("from {} to {}".format(url, target_filename))

            r = self.check_url(url, target_dir=self.target_dir)

            total_size = int(r.headers.get("content-length", 0))
            pbar = tqdm(total=total_size, unit="iB", unit_scale=True)
            with open(target_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        pbar.update(self.chunk_size)
                        f.write(chunk)
            pbar.close()

        else:
            if self.verbose:
                print("{} available locally, skip downloading".format(target_filename))

    @staticmethod
    def check_url(url, target_dir=None):
        r = requests.get(url, target_dir, stream=True)
        if r.status_code != 200:
            print("{} not available".format(url))
            sys.exit()

        else:
            return r
