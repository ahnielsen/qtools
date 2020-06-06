Introduction
============

Overview
--------

Qtools is a Python package developed specifically for numerical analysis in the context of earthquake engineering. It contains classes and functions for computation, processing and plotting of time histories, response spectra and power spectra. Qtools defines three new classes, :class:`qtools.ResponseSpectrum`, :class:`qtools.TimeHistory` and :class:`qtools.PowerSpectrum`, and several new functions.

Getting Started
---------------

Place the Qtools package (i.e. the folder named "qtools") in your PYTHONPATH. Then import Qtools:

   >>>import qtools as qt
   
> Note: In the examples given within the documentation, Qtools is always imported as `qt`, and NumPy is always imported as `np`.  

Prerequisites
-------------

Qtools relies on the following packages in addition to standard packages such as :mod:`math`, :mod:`copy` and :mod:`itertools`:

* NumPy
* SciPy
* Matplotlib
* PyRVT (optional)

> Note: If the PyRVT package is not present on your system, calling the function `qtools.rs2ps()` will have no effect. 

Author
------

Andreas H. Nielsen<br>
Principal Engineer, **Atkins**

License
-------

This project is not yet licensed.

