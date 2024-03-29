# NFFT4ANOVA
### Source code for our Paper ["Learning in High-Dimensional Feature Spaces Using ANOVA-Based Matrix-Vector Multiplication"](https://arxiv.org/abs/2111.10140)

This package uses the [FastAdjacency](https://github.com/dominikalfke/FastAdjacency) package by Dominik Alfke to perform NFFT-based fast summation to speed up kernel-vector multiplications for the ANOVA kernel. It is targeted at large-scale kernel evaluations. We demonstrate our method's computational power by using it for kernel ridge regression, which is just one of many possible applications. For more details, see the above-mentioned paper. A huge benefit of this package is that even for very large-scale data, all codes can easily be run on a standard laptop computer in absolutely reasonable time, so that no superior hardware is required.


# Installation
- This software has been tested with Python 3.8.
- This software depends on Alfke's FastAdjacency Package. We refer to https://github.com/dominikalfke/FastAdjacency#readme for installation instructions.


# Usage

This package consists of the following three classes:
- `kernel_vector_multiplication` compares the standard kernel-vector multiplication with kernel-vector multiplication with NFFT-based fast summation in runtime and approximation error.
- `NFFTKernelRidge` performs NFFT-based kernel ridge regression.
- `GridSearch` searches on candidate parameter values for one of the classifiers `NFFTKernelRidge`, `sklearn KRR`or `sklearn SVC`.

See [`test/showcase_kernel_vector_multiplication.ipynb`](https://github.com/wagnertheresa/NFFT4ANOVA/blob/main/test/showcase_kernel_vector_multiplication.ipynb) and [`test/showcase_nfft_krr.ipynb`](https://github.com/wagnertheresa/NFFT4ANOVA/blob/main/test/showcase_nfft_krr.ipynb) for an example.
