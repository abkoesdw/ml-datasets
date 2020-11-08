from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as fh:
    long_description = fh.read()

setup(
    name="ml-datasets",
    version="0.0.1",
    author="Arief Koesdwiady",
    author_email="ariefbarkah@gmail.com",
    description="This is the collections of datasets for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "requests>=2.24.0",
        "tqdm>=4.46.1",
        "idx2numpy==1.2.2",
        "matplotlib>=3.2.2",
        "rdata>=0.2.1",
        "ipykernel==5.3.4",
    ],
    license="GNU General Public License v3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
)
