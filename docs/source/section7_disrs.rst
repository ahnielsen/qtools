Direct generation of in-structure response spectra (DISRS)
==========================================================

Overview
--------

DISRS can be executed as a standalone script or as function in the Qtools package.
The code implements the method proposed by `Jiang et al. (2015)`_
and `Li et al. (2015)`_ with some modification to the calculation of the
so-called t-response spectrum.
The code estimates the ISRS for a 1-DoF system attached to a N-DoF system.
The code assumes that the third direction (du = 2) is the vertical direction.
A complete documentation is available as a standalone PDF document
(EN/NU/TECH/TGN/031).

Calling DISRS as a function
---------------------------

For function documentation, see below.

References
----------
.. _Jiang et al. (2015):

Jiang, W; Li, B; Xie W-C; & Pandey, M.D., 2015: "Generate floor response spectra: Part 1.
Direct spectra-to-spectra method", *Nuclear Engineering and Design*, **293**, pp. 525-546
(http://dx.doi.org/10.1016/j.nucengdes.2015.05.034)

.. _Li et al. (2015):

Li, B; Jiang, W; Xie W-C; & Pandey, M.D., 2015: "Generate floor response spectra, Part 2:
Response spectra for equipment-structure resonance", *Nuclear Engineering and Design*, 
**293**, pp. 547-560 (http://dx.doi.org/10.1016/j.nucengdes.2015.05.033)

.. autofunction:: qtools.disrs

