import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boolem",
    version="0.0.5",
    author="Lifan Liang",
    author_email="lil115@pitt.edu",
    description="Boolean matrix factorization on RNA expression data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LifanLiang/EM_BMF",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)
