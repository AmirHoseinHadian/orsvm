.. ORSVM documentation master file, created by
   sphinx-quickstart on Mon Sep 20 11:31:05 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ORSVM's documentation!
=================================

`orsvm` is a free software package which provides a **SVM** classifier with  some novel orthogonal polynomial kernels. This library provides a complete path of using the SVM classifier from **normalization** to calculation of **SVM equation** and the final **evaluation**.
Convext optimization is done using `cvxopt`_, which solves teh conext SVM's equation and returns the support vectors as result.

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
   :maxdepth: 1
   :caption: How to:

   notebooks/How_to_use_orsvm.ipynb
   
.. toctree::
   :maxdepth: 1
   :caption: Examples:

   notebooks/Chebyshev_fitting
   notebooks/Legendre_fitting
   notebooks/Gegenbauer_fitting
   notebooks/Jacobi_fitting   
   
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
