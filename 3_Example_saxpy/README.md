# 3. SAXPY

This folder contains four implementations of [SAXPY], a function that
combines scalar multiplication with vector addition. It takes as input
two vectors of 32-bit floats `X` and `Y` with `N` elements each, and a
scalar value `A`. It multiplies each element `X[i]` by `A` and adds the
result to `Y[i]`.

## a_sequential.py

This is a simple, sequential implementation of SAXPY in plain Python.
Performance is not spectacular since it runs on the Python interpreter.

## b_numpy.py

This implementation uses [NumPy], a Python library that provides
mathematical objects and operations. It is the fundamental package
for scientific computing in Python. It is partially written in C to
improve performance.

## c_numba.py

This implementation uses [Numba], a JIT compiler that can translate
a subset of Python and NumPy code into fast machine code at runtime.

Note that, since Numba's JIT compiler is "lazy" by default (i.e. it
only runs "just in time", right before the compiled code needs to be
executed for the first time), we need to manually force the JIT compiler
to run before measuring execution time.

## main.cpp

This is a C++ implementation that uses Intel's [OneAPI TBB], a flexible
performance library that simplifies the work of adding parallelism to
complex applications across accelerated architectures.

[SAXPY]: https://developer.nvidia.com/blog/six-ways-saxpy/
[NumPy]: https://numpy.org/
[Numba]: https://numba.pydata.org/
[OneAPI TBB]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html
