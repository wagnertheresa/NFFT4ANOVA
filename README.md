# NFFT4ANOVA
### Source code for our Paper "Learning in High-Dimensional Feature Spaces Using ANOVA-Based Matrix-Vector Multiplication"

This package uses the [FastAdjacency](https://github.com/dominikalfke/FastAdjacency) package by Dominik Alfke to perform NFFT-based fast summation to speed up kernel-vector multiplications for the ANOVA kernel. This method is targeted at large-scale kernel evaluations. For more details, see the above-mentioned paper. A huge benefit of this package is that even for very large-scale data, all codes can easily be run on a standard laptop computer in absolutely reasonable time, so that no superior hardware is required.


# Installation
- This software has been tested with Python 3.8.
- This software depends on Alfke's FastAdjacency Package. We refer to https://github.com/dominikalfke/FastAdjacency#readme.


# Usage

This package consists of the following three classes:
- jhdvf
- djaskvg

See ->showcase and ->showcase for an example.
