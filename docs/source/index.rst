.. orsvm documentation master file, created by
   sphinx-quickstart on Thu Nov 24 00:07:17 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to orsvm's documentation!
=================================

`orsvm` is a free software package which provides a **SVM** classifier with  some novel orthogonal polynomial kernels. This library provides a complete path of using the SVM classifier from **normalization** to calculation of **SVM equation** and the final **evaluation**.
Convext optimization is done using `cvxopt`_, which solves the convext SVM's equation and returns the support vectors as the result. `orsvm` benefits from the orthogonal kernels' capabilities to constitute the kernel matrix, which is thereafter used by `cvxopt` to as one of input matrices , to form the SVM equation. Another novelty in `orsvm` is that it is now possible to transform the dataset to a fractional space, as well as a normal space, a.k.a normalization.
For a comprehensive introduction to fractional orthogonal kernel function refer to `Learning with Fractional Orthogonal Kernel Classifiers in Support Vector Machines`_ book.

.. _Learning with Fractional Orthogonal Kernel Classifiers in Support Vector Machines: https://link.springer.com/book/9789811965524
.. _cvxopt: https://cvxopt.org/

.. image:: _static/identicons.png

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   installation
   references
   credits

.. toctree::
   :maxdepth: 2
   :caption: How to:

   _notebooks/How_to_use_orsvm.ipynb
   _notebooks/How_to_use_prefitted_model.ipynb
   
.. toctree::
   :maxdepth: 2
   :caption: Examples:
   
   _notebooks/Legendre_fitting.ipynb
   _notebooks/Chebyshev_fitting.ipynb
   _notebooks/Gegenbauer_fitting.ipynb
   _notebooks/Jacobi_fitting.ipynb


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
