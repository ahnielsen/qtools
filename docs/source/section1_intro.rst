Introduction
============

Overview
--------

Qtools is a Python package developed specifically for numerical analysis in the context of earthquake engineering. It contains classes and functions for computation, processing and plotting of time histories, response spectra and power spectra. Qtools defines six new classes, :class:`qtools.ResponseSpectrum`, :class:`qtools.EnergySpectrum`, :class:`qtools.TimeHistory`, :class:`qtools.TimeHistorySet`, :class:`qtools.PowerSpectrum` and :class:`qtools.FourierSpectrum`, and several new functions.

Getting Started
---------------

Place the Qtools package (i.e. the folder named "qtools") in your PYTHONPATH. Then import Qtools::

   >>>import qtools as qt
   
.. note::

   In the examples given within this documentation, Qtools is always imported as `qt`, and NumPy is always imported as `np`.  

Prerequisites
-------------

Qtools relies on the following packages in addition to standard packages such as :mod:`math`, :mod:`copy` and :mod:`itertools`:

* :py:mod:`NumPy`
* :py:mod:`SciPy`
* :py:mod:`Matplotlib`
* :py:mod:`PyRVT` (optional)

.. note::

   If the :mod:`PyRVT` package is not present on your system, calling the function :func:`qtools.rs2ps` will have no effect. 

Author
------

| Andreas H. Nielsen
| Principal Engineer, **Atkins**
| andreas.nielsen@atkinsglobal.com

License
-------
Copyright (C) 2020 Andreas H. Nielsen

The Qtools package is released under the GNU General Public License Version 3 or later. 

Qtools is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Qtools is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Qtools (in the file LICENSE.txt). If not, see <https://www.gnu.org/licenses/>.