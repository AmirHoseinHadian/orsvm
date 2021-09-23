ORSVM is a free software package which provides a SVM classifier with some novel orthogonal polynomial kernels. This library provides a complete chain of using the SVM classifier from normalization to calculation of SVM equation and the final evaluation. But please note that, there are some necessary steps before normalization, that should be handled for every data-set, such as duplicate checking, Null values or outlier handling, or even dimensionality reduction or whatever enhancements that may apply to a data-set. These steps are out of scope of SVM algorithm, thereupon, orsvm package. Instead, normalization step which is a must before sending data points into kernels, is handled directly in orsvm, by calling the relevant function.



Installation 
============

You can install the osvm package using::
``pip install orsvm``, 

or get it directly from `GitHub`_ and install it locally using:: 

``pip install orsvm.zip``.


.. _Github: https://gitlab.com/mohammad.akhavan75/osvm

Dependencies
------------
Following dependencies will b eautomatically installed through `` pip install orsvm``. Still one can install it separately.

- numpy
- cvxopt
- sk_learn
- pandas


Conda environment (suggested)
-----------------------------

If you have Andaconda or miniconda installed and you would like to create a separate environment for the osvm package, do the following::

	conda create --n ORSVM python=3.8
	conda activate ORSVM
	pip install orsvm
