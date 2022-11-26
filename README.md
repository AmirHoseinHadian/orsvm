![LOGO](docs/_static/identicons.png)

# ORSVM 

`ORSVM` is a free software package which provides a SVM classifier with  some novel orthogonal polynomial kernels.
This library provides a complete path of using the SVM classifier from normalization to calculation of SVM equation and the final evaluation.
In order to classify the dataset with ORSVM, there is a need to normalize the dataset whether using normal or fractional kernels.
ORSVM library needs numpy and cvxopt libraries to be installed. Arrays, matrices and linear algebraic functions have been used repeatedly from numpy and
 the heart of SVM algorithm which is finding the Support Vectors is done by use of a convex quadratic solver from cvxopt library which is in turn a free python package for 
 convex optimization. 
 For a comprehensive introduction to fractional orthogonal kernel function and the use cases in SVM, refer to [Learning with Fractional Orthogonal Kernel Classifiers in Support Vector Machines](https://link.springer.com/book/9789811965524) book.
 
A suitable guide on `cvxopt` package is available at http://cvxopt.org about installation and how to use.
 

 ## Install
 You can install orsvm using:
```
pip install orsvm
```

### Dependencies
Following dependencies will be installed:
- cvxopt
- pandas
- numpy
- sklearn

### Conda environment (suggested)
```
conda create --n ORSVM python=3.8 pandas sklearn numpy cvxopt
conda activate ORSVM
pip install orsvm
```

## Documentation
The latest documentation can be found here: http://orsvm.readthedocs.io/

## Cite
[![DOI](https://zenodo.org/badge/409558175.svg)](https://zenodo.org/badge/latestdoi/409558175)
