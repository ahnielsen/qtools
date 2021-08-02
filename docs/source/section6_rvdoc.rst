Random vibration tools
======================

The peak factor function class
------------------------------

.. autoclass:: qtools.PeakFactorFunction
   :members:
   
Conversion functions
--------------------

Two functions are provided for spectral conversions: :func:`qtools.convert2ps`, which converts a response spectrum to a compatible power spectrum, and :func:`qtools.convert2rs`, which convert a power spectrum to a compatible response spectrum. Each function requires an instance of :class:`qtools.PeakFactorFunction` as argument.

.. autofunction:: qtools.convert2ps

.. autofunction:: qtools.convert2rs

Internal conversion algorithms
------------------------------

The following functions are called by :func:`qtools.convert2ps` and :func:`qtools.convert2rs`. Their docstrings are included for information only.

.. autofunction:: qtools.random_vibration.calc_comp_ps

.. autofunction:: qtools.random_vibration.calc_comp_Var