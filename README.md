Approximating Orthogonal Matrices with Effective Givens Factorization
================
This is an implementation of the algorithms analyzed in our ICML 2019 paper:

**Approximating Orthogonal Matrices with Effective Givens Factorization** (Thomas Frerix, Joan Bruna; ICML 2019)

The core algorithm is written in C++/CUDA and interfaces with python through pybind11.

Installation
-------------------
1. Make sure you have a running Python 3 (tested with Python 3.7) ecosytem, e.g. through conda, and an Nvidia GPU (tested with CUDA 9.0 on a Titan X).
2. Install the python dependencies via `pip install -r pip_requirements.txt`.
3. Install [CUB](https://nvlabs.github.io/cub/) (tested with 1.8.0) and [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) (tested with 3.2). 
4. Specify the environment variables `CUB_INCLUDE_DIR` to the CUB library and `PYBIND_INDLUDE_DIR` to the python library path of your installation that includes pybind11 (installed as a python package).
5. Run `cmake .` and `make` to compile the code.
6. Run the end-to-end tests by calling `pytest test_givens_gpu.py -v -s`.

Example
-------------------
The file `example.py` contains an example of factorizing an orthogonal matrix with various algorithms, which are explained and discussed in the paper.
