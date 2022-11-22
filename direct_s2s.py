"""
..
	Package: Qtools
	Module: direct_s2s
	(C) 2020-2022 Andreas H. Nielsen
	See README.md for further details.

Overview
--------
The DirectS2S module estimates the in-structure response spectra (ISRS) for a
secondary 1-DoF system attached to a primary N-DoF systemt using a direct
spectrum-to-spectrum method.

The module implements the following methods:

	* `Jiang et al. (2015)`_ and `Li et al. (2015)`_ with some modification
	  to the calculation of the so-called t-response spectrum.
	* `Der Kiureghian et al. (1981)`_.

The first method requires special treatment of the vertical direction.
Therefore, in the implementation of the first method, it was assumed that
the third direction is the vertical direction.
In the second method, no distinction between horizontal and vertical
excitation is required, and so the vertical direction can be any of the three
principal diretions.
A more complete documentation is available as a standalone PDF document.

.. caution::

	None of the implemented methods are suitable for systems with high damping
	values (greater than, say, 10%).
	The Der Kiureghian (1981) method is not suitable for systems with closely
	spaced modes.

Using the module
----------------
The module provides one entry function called :func:`DirectS2S`. There are
two ways to provide the required response spectra and modal data to the
function:

1. Define two lists as follows:

	a. A list of lists of response spectra (`rslist`). If the number of
	   excitation directions is `ND`, then `rslist` must have at least `ND`
	   sublists and each sublist must contain at least one response
	   spectrum. Each sublist defines the spectral accelerations at one or more
	   damping ratios. For example, `rslist` could be defined as::

		rslist = [[rsx1,rsx2,rsx3],[rsy1,rsy2,rsy3],[rsz1,rsz2,rsz3]]

	   where `rsx1` is the input spectrum in the x-direction at damping level
	   1, `rsx2` is the input spectrum in the x-direction at damping level 2,
	   and so on. The response spectra in each sublist must be arranged so
	   that the damping level is increasing from left to right.
	   All response spectra must be defined at the same frequencies (the class
	   method :meth:`.ResponseSpectrum.interp` can be used for this purpose).
	   The damping levels must be consistent from one sublist to the next so
	   that the damping level of `rsx1` is the same as the damping level of
	   `rsy1`, which again is the same as the damping level of `rsz1`, and so
	   on.
	b. A list of arrays (`mdlist`) that defines the modal properties of
	   the primary system, and where ``mdlist[0]`` is a 1D array with
	   natural frequencies (with ``Np = len(mdlist[0])``);
	   ``mdlist[1]`` is a 1D array with modal damping ratios;
	   ``mdlist[2]`` is a 2D array with participation factors (the shape of
	   this array must be ``(Np,ND)`` or just ``(Np,)`` if ``ND == 1``);
	   ``mdlist[3]`` is a 2D array with modal displacements (the shape of this
	   array must be ``(Np,ND)`` or just ``(Np,)`` if ``ND == 1``).

2. If `rslist` is not provided, then the function will look for
   the file `SpectralData.txt` in the specified working directory. Likewise,
   if `mdlist` is not provided, then the function will look for
   the file `ModalData.txt` in the specified working directory. For more
   information about the format of `SpectralData.txt` and `ModalData.txt`,
   see the standalone DirectS2S documentation.

To execute DirectS2S, follow these instructions:

	* Install Qtools somewhere in the local PYTHONPATH.
	* Create the two lists `rslist` and `mdlist`. Alternatively, create and
	  save files `ModalData.txt` and `SpectralData.txt` in a suitable working
	  directory.
	* Copy the sample code provided below and save in a main
	  script. Modify the user-defined parameters as required.
	* Execute the main script from the command line or through an IDE
	  application.

Sample main script
------------------

The following is a typical main script for execution of DirectS2S (assuming
`SpectralData.txt` and `ModalData.txt` have been created and saved in the
working directory)::

	import qtools as qt

	# ------------------------------------------
	#          USER-DEFINED PARAMETERS
	# ------------------------------------------
	# Path to working directiory (absolute or relative)
	wd = '../directS2S/Validation/Case 1/'

	# Method {'Jiang (2015)', 'Der Kiureghian (1981)'}
	method = 'Jiang (2015)'

	# Number of excitation directions to be taken into account: {1, 2, 3}
	ND = 3

	# Define output direction of interest: {0, 1, 2}, with 2 being vertical.
	OD = 2

	# Define the desired damping ratio of the ISRS
	z0 = 0.05

	# Lower bound damping value (for avoidance of division by zero in case a
	# primary mode has been defined with zero damping)
	zmin = 0.001

	# Include residual response?
	INRES = True

	# -----------------------------------------------
	# SPECIFIC PARAMETERS REQUIRED FOR 'Jiang (2015)'
	# -----------------------------------------------
	# Specify ground type ('H' for hard, 'S' for soft; only relevant for
	# vertical excitation, OD = 2)
	GT = 'H'

	# Define corner frequencies (in Hz)
	fc = (12,40)

	# --------------------------------------------------------
	# SPECIFIC PARAMETERS REQUIRED FOR 'Der Kiureghian (1981)'
	# --------------------------------------------------------
	# Specify mass of secondary system (in kg or other consistent unit)
	m0 = 300.0

	# ------------------------------------------
	#       END OF USER-DEFINED PARAMETERS
	# ------------------------------------------

	# Call the main function
	rs = qt.directS2S(ND, OD, z0, method=method, zmin=zmin, INRES=INRES,
	               fc=fc, GT=GT, wd=wd, m0=m0)

References
----------
.. _Der Kiureghian et al. (1981):

	A Der Kiureghian, JL Sackman & B Nour-Omid, 1981: *Dynamic Response of
	Light Equipment in Structures*, Report No. UBC/EERC-81/05, Earthquake
	Engineering Research Centre, University of California, Berkeley.

.. _Jiang et al. (2015):

	W Jiang, B Li, W-C Xie & MD Pandey, 2015: "Generate floor response spectra,
	Part 1: Direct spectra-to-spectra method", Nuclear Engineering and Design,
	**293**, pp. 525-546 (http://dx.doi.org/10.1016/j.nucengdes.2015.05.034).

.. _Li et al. (2015):

	Li, B; Jiang, W; Xie W-C; & Pandey, M.D., 2015: "Generate floor response
	spectra, Part 2: Response spectra for equipment-structure resonance",
	*Nuclear Engineering and Design*, **293**, pp. 547-560
	(http://dx.doi.org/10.1016/j.nucengdes.2015.05.033).

"""

# TO-DO:
# Implement option to choose vertical direction as sole excitation direction
# Only important for Jiang et al method.

import numpy as np
import time
from pathlib import Path
from qtools.response_spectrum import ResponseSpectrum
from qtools.config import Info, version, set_module
from qtools import ds2smods as dsm

@set_module('qtools')
def directS2S(ND, OD, z0, method='Jiang (2015)', zmin=0.001, INRES=True,
			  **kwargs):
	"""Compute an in-structure response spectrum (ISRS) using a direct
	spectrum-to-spectrum method. The named parameters supported by the
	``**kwargs`` container are listed under **Other parameters** below.

	Parameters
	----------
	ND : int, {1, 2, 3}
		Number of excitation directions to take into account.
	OD : int, {0, 1, 2}
		Direction of in-structure response ('output direction').
	z0 : float
		Damping ratio of the secondary system.
	method : str, optional
		The numerical method used to compute the ISRS. Valid options are
		'Jiang (2015)' (default) and 'Der Kiureghian (1981)'.
	zmin : float, optional
		Lower bound modal damping value of the primary system.
		Useful if some primary modes have zero or unrealistically low damping.
		Default 0.001.
	INRES : bool, optional
		Include residual response in the computation. Default `True`.
		See DISRS documentation for further information.
	wd : str, optional
		Path to working directory. If this is not provided, `wd` will default
		to the current working directory.

	Other parameters
	----------------
	rslist : a list of lists of response spectra, optional
		Input spectra for ISRS computation. If `rslist` is not provided,
		the function will look for a file named `SpectralData.txt` in the
		working directory.
		The length of `rslist` must satisfy: ``len(rslist) >= ND``.
		The length of ``rslist[0]`` determines the number of damping values
		(`Nzb`).
		If ``rslist[1]`` is provided, then ``len(rslist[1]) == len(rslist[0])``.
		If ``rslist[2]`` is provided, then ``len(rslist[2]) == len(rslist[0])``.
	mdlist : a list of arrays, optional
		This parameter contains the modal properties of the primary system,
		where:

		* ``mdlist[0]`` is a 1D array with natural frequencies
		  (with ``Np = len(mdlist[0])`` being the number of modes);
		* ``mdlist[1]`` is a 1D array with modal damping ratios
		  (the shape of this array must be (`Np`,));
		* ``mdlist[2]`` is a 2D array with participation factors
		  (the shape of this array must be (`Np`, `ND`));
		* ``mdlist[3]`` is a 1D or 2D array with modal displacements
		  (the shape of this array must be (`Np`, `ND`) or just (`Np`,)
		  if `ND` = 1).

		Lists and tuples are acceptable types of arrays. The function will
		convert array-like objects into ndarrays.
	fc : tuple of floats, required
		Corner frequencies. Only used with the Jiang (2015) method. See the
		DirectS2S documentation for further information.
	GT : str, optional
		Ground type ('H' for hard, 'S' for soft). Only relevant for vertical
		excitation. Only used with the Jiang (2015) method. Default 'H'.
	m0 : float, optional
		Mass of secondary system. Only used with the Der Kiureghian (1981)
		method. Default value 0.

	Returns
	-------
	rs : an instance of class ResponseSpectrum
		The in-structure response spectrum for response in direction `OD`.

	Examples
	--------
	The following definition of `mdlist` is appropriate for a single
	degree-of-system with frequency `f0` and damping ratio `z0`::

		mdlist = [[f0], [z0], [[1]], [1]]

	Notes
	-----
	A more thorough DirectS2S documentation is available as a PDF document:
	EN/NU/TECH/TGN/031, Direct Generation of In-Structure Response Spectra.
	"""
	# Time the execution
	start = time.perf_counter()
	Info.note('Computing in-structure response spectrum with method: {}'
			   .format(method))

	# Check if a path has been specified; if not, assume current working directory
	if 'wd' in kwargs:
		wd = Path(kwargs['wd'])
	else:
		wd = Path.cwd()

	# Retrieve modal information (from textfile if not supplied in **kwargs)
	if 'mdlist' in kwargs:
		mdlist = kwargs['mdlist']
		fp = np.asarray(mdlist[0])
		zp = np.fmax(mdlist[1],zmin)
		gam = np.asarray(mdlist[2])
		phi = np.asarray(mdlist[3])
	else:
		MD = np.loadtxt(wd / 'ModalData.txt')
		fp = MD[:,0]
		zp = np.fmax(MD[:,1],zmin)
		gam = MD[:,2:(2+ND)]
		phi = MD[:,(2+ND+OD)]

	# Retrieve response spectra at the base of the primary system.
	# If rslist is not provided, read spectral input data from file
	if 'rslist' in kwargs:
		rslist = kwargs['rslist']
		assert len(rslist) >= ND, ('The length of rslist is insufficient to '
							 'compute response over {} directions'.format(ND))
		Nfb = rslist[0][0].ndat
		Nzb = len(rslist[0])
		fb = rslist[0][0].f
		zb = np.empty(Nzb)
		SAb = np.empty((Nfb,Nzb,ND))
		for i in range(Nzb):
			zb[i] = rslist[0][i].xi
			for j in range(ND):
				SAb[:,i,j] = rslist[j][i].sa
	else:
		SDb = np.loadtxt(wd / 'SpectralData.txt')
		Nfb = np.size(SDb,axis=0)-1
		Nzb = (np.size(SDb,axis=1)-1)//ND
		fb = SDb[1:Nfb+1,0]
		zb = SDb[0,1:Nzb+1]
		SAb = np.empty((Nfb,Nzb,ND))
		for d in range(ND):
			SAb[:,:,d] = SDb[1:Nfb+1,d*Nzb+1:(d+1)*Nzb+1]

	if method=='Jiang (2015)':
		if 'fc' in kwargs:
			fc = kwargs['fc']
		else:
			raise ValueError('With method=\'Jiang (2015)\', the parameter fc '
					'must be defined and provided.')
		if 'GT' in kwargs:
			GT = kwargs['GT']
		else:
			GT = 'H'
		SA_ISRS = dsm.j15_main(SAb, fb, zb, fp, zp, gam, phi, z0, OD, ND,
						 INRES, fc, GT)

	elif method=='Der Kiureghian (1981)':
		if 'm0' in kwargs:
			m0 = kwargs['m0']
		else:
			m0 = 0.
		SA_ISRS = dsm.dk81_main(SAb, fb, zb, fp, zp, gam, phi, m0, z0, OD, ND,
						  False)

	else:
		raise ValueError('Method {} is not supported.'.format(method))

	rs = ResponseSpectrum(fb, SA_ISRS, xi=z0,
					   label='ISRS {}% Dir {}'.format(100*z0, OD))

	# Write output to file
	outfil = Path(wd) / 'DirectS2S_Output_{}.txt'.format(OD)
	localtime = time.asctime(time.localtime(time.time()))
	head1 = ('# In-structure response spectrum computed by Qtools '
		  'v. {} on {}'.format(version, localtime))
	head2 = ('# Method = {}, direction = {}, damping ratio = {}, '
		  'residual response {}'.format(method, OD, z0, 'included' if INRES
								 else 'excluded'))
	head3 = '#'
	if method=='Der Kiureghian (1981)':
		head3 = '# Secondary system mass = {}'.format(m0)
	header = '\n'.join([head1,head2,head3])
	np.savetxt(outfil, np.array([rs.f,rs.sa]).T, header=header)

	# End timing
	stop = time.perf_counter()
	Info.note('Time to execute (min): {:6.2f}'.format((stop-start)/60))
	Info.end()

	return rs