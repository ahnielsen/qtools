Introduction
============

Overview
--------

Qtools is a Python package developed specifically for numerical analysis in the context of 
earthquake engineering. It contains classes and functions for computation, processing and plotting 
of time histories, response spectra and power spectra. Qtools defines six new classes, 
`qtools.ResponseSpectrum`, `qtools.EnergySpectrum`, `qtools.TimeHistory`, `qtools.TimeHistorySet`, 
`qtools.PowerSpectrum` and `qtools.FourierSpectrum`, and several new functions.

Getting Started
---------------

Place the Qtools package (i.e. the folder named "qtools") in your PYTHONPATH. Then import Qtools:

    import qtools as qt

> Note: In the examples given within the documentation, Qtools is always imported as `qt`, and 
  NumPy is always imported as `np`.

Prerequisites
-------------

Qtools relies on the following packages in addition to standard packages such as `math`, `copy` 
and `itertools` (version numbers correct at the completion of Qtools version 2.0):

* NumPy (1.18.1)
* SciPy (1.4.1)
* Matplotlib (3.1.3)
* Numba (0.48.0)

An update to the latest versions of the above packages is planned during 2021.

> Note: The Fortran subroutine `qt.dll.auxforsubs` has been compiled for Windows x64 systems only. 
  On other systems Qtools should automatically ignore the Fortran code and replace it with slower 
  Python code. This affects `qt.calcrs` only.

Documentation
-------------

A documentation is available in the ./docs folder. However, this has to be compiled with Sphinx
for HMTL / PDF viewing. The current documentation has been compiled with Sphinx 2.4.0.

Version numbering
-----------------

Versions are numbered using two digits separated by a full stop: *m.n*. All versions with the same 
*m* number should be backwards compatible and give the same results. Versions with higher *n*
number are those with minor additions, clarifications, efficiency improvements, etc. The version
number is available as `qt.__version__`.

Versions
--------

### Version 2.2 (March 2021)

- Improvements were made to the numerical stability of the function `qt._intersection`, which in
  turn serves `qt.envelope` and `qt.peakbroad`. In previous versions, response spectra with nearly
  parallel line segments or line segments intersecting close to one of the existing frequency
  points could cause numerical instability in the function.
- Minor improvements to `qt.plotrs`.

### Version 2.1 (March 2021)

- Multiple updates to the two plotting functions `qt.plotrs` and `qt.plotps` providing greater user
  control over appearance and style.
- A power spectrum is now defined purely as a one-sided smoothed function of frequency. In
  principle, this constitutes a major version update. However, it is surmised that the older
  implementation (in which a power spectrum was principally defined as a double-sided smoothed or
  unsmoothed function) was not used; therefore, this change is included in a minor version update.
- New methods `qt.PowerSpectrum.moment` and `qt.PowerSpectrum.moments` for computation of spectral
  moments.

### Version 2.0 (February 2021)

- The module `qt.disrs` was replaced by a new module named `qt.direct_s2s`. This now provides all
  functions required to compute in-structure response spectra using direct spectrum-to-spectrum
  methods. The interface with `qt.direct_s2s` is now through a single function `qt.directS2S()`.
- The original direct spectrum-to-spectrum method contained in `qt.disrs` (Jiang et al. (2015)) was
  re-written, and the speed of the implementation was improved.
- A new method (Der Kiureghian (1981)) was added to `qt.direct_s2s`. For full references see the
  documentation.
- There is still a stand-alone documentation for `qt.direct_s2s`, which is under development.
  However, a working documentation has also been added to the main Qtools documentation.
- In `qt.plotps`, grids are now added to all plots.

### Version 1.2 (January 2021)

- Docstring for class `qt.TimeHistorySet` written.
- Added arguments `label` and `fmt` to functions `qt.meanrs` and `qt.loadrs`.

### Version 1.1 (January 2021)

- Error corrected in function `qt.plotps` (function updated to accommodate new definition of
  `unit` attribute
  introduced in version 1.0).
- Function `qt.calcps` now determines the unit of the power spectrum.
- Function `qt.envelope` now takes as its first argument a list containing any number of response 
  spectra (`qt.envelope(rslist)`). The old call signature (`qt.envelope(rs1,rs2)`) will still
  work; however, this signature is deprecated and will become obsolete in version 2.0.
- The `option` argument is no longer used for anything in `qt.envelope`.
- New argument `mutate` added to `qt.ResponseSpectrum.interp`.
- New function `qt.meanrs` added.

### Version 1.0 (November 2020)

- Error corrected in function `qt.calcrs` (affected cases for which the maximum frequency `fmax` 
  was greater than the Nyquist frequency of the input time history `th.fNyq`).
- Function `qt.interpft` added for interpolation of time histories.
- Attribute `ei` (meant to contain input energy) removed from `qt.ResponseSpectrum`. As a 
  replacement, a new class `qt.EnergySpectrum` is introduced.
- Fourier amplitudes removed from class `qt.PowerSpectrum`.
- New class `qt.FourierSpectrum` with accompanying creator function `qt.calcfs`.
- Parameter `color` added to `qt.ResponseSpectrum.setLineFormat`.
- Minor change to the criterion for removal of duplicate frequencies in
  `qt.ResponseSpectrum.interp`.
- Added class method `qt.TimeHistory.differentiate`.
- In `qt.loadth`, the Nyquist frequency is now rounded to two digits from the decimal point. This 
  was done as a pragmatic way to deal with situations where `th.fNyq` was nearly, but not quite
  equal to a whole number due to finite precision in input data.
- New argument `truncate` added to function `qt.peakbroad`.
- New function `qt.calcrs_cmp` added.
- New function `qt.dmpinterp` added.
- New class `qt.TimeHistorySet` added.

### Version 0.2 (June 2020)

First fully documented version.


Author
------

Andreas H. Nielsen<br>
Principal Engineer, **Atkins**

License
-------
Copyright (C) 2020-2021 Andreas H. Nielsen

The Qtools package is released under the GNU General Public License Version 3 or later.

Qtools is free software: you can redistribute it and/or modify it under the terms of the GNU 
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Qtools is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General 
Public License for more details.

You should have received a copy of the GNU General Public License along with Qtools (in the file 
LICENSE.txt). If not, see <https://www.gnu.org/licenses/>.

