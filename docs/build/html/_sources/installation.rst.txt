ORSVM is a free software package that provides an SVM classifier with some novel orthogonal polynomial kernels. This library provides a complete chain of using the SVM classifier from normalization to calculation of the SVM equation and the final evaluation. But please note that there are some necessary steps before normalization, that should be handled for every data set, such as duplicate checking, Null values or outlier handling, or even dimensionality reduction or whatever enhancements may apply to a data set. These steps are out of the scope of the SVM algorithm, thereupon, the orsvm package. Instead, the normalization step which is a must before sending data points into kernels is handled directly in orsvm, by calling the relevant function



Installation 
============

You can install the osvm package using::

	pip install orsvm

or get it directly from `GitHub`_ and then install it locally using the following command:: 

	pip install orsvm.zip


.. _Github: https://github.com/AmirHoseinHadian/orsvm

Dependencies
------------
Following dependencies will be automatically installed through `pip install orsvm`. Still one can install them separately.

- numpy
- cvxopt
- scikit_learn
- pandas


Conda environment (suggested)
-----------------------------

If you have Andaconda or miniconda installed and you would like to create a separate environment for the osvm package, do the following::

	conda create --n ORSVM python=3.10
	conda activate ORSVM
	pip install orsvm
