Response spectra
================

The response spectrum class
---------------------------

.. autoclass:: qtools.ResponseSpectrum
   :members:

Response spectrum creation
--------------------------

.. autofunction:: qtools.calcrs

.. autofunction:: qtools.calcrs_cmp

.. autofunction:: qtools.loadrs

Design response spectrum creation
---------------------------------

.. autofunction:: qtools.ec8_2004

.. autofunction:: qtools.dec8_2017

.. autofunction:: qtools.ieee693_2005

.. autofunction:: qtools.pml

.. autofunction:: qtools.eur_vhr

Derived response spectrum computation
-------------------------------------

.. autofunction:: qtools.meanrs

.. autofunction:: qtools.envelope

.. autofunction:: qtools.peakbroad

.. autofunction:: qtools.dmpinterp

See also :meth:`.ResponseSpectrum.interp`.