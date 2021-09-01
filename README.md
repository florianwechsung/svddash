# SVDdash

A simple implementation in C++ with Python bindings for the derivative of

    f(A) = \sum_{i=1}^d phi( alpha*(\sigma-threshold)^+ )

where

    phi(t) = 0.5 * pow(std::cosh(t)-1, 2);

(other penalties can easily be swapped in).

Functions of this form are a useful measure for mesh quality in shape optimisation.

## Usage

    import svddash
    import numpy

    numpy.random.seed(1)
    d = 5
    A = numpy.random.standard_normal(size=(d, d))

    threshold = 1.
    alpha = 3.
    f, df, d2f = svddash.compute(A, threshold, alpha)

After running this, `f` is a scalar, `df` is a `dxd` matrix, and `d2f` is a `dxdxdxd` tensor.

## Installation

*IMPORTANT*: You need to clone the repository recursively in order to get all the required submodules.

    git clone --recursive https://github.com/florianwechsung/svddash.git
    pip3 install -e .
    python3 examples/example.py
