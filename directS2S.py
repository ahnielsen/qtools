"""
Package: Qtools
Module: directS2S
(C) 2020-2021 Andreas H. Nielsen
See README.md for further details.

Overview
--------
This module can be executed as a standalone script or as function in the Qtools
package. The script relies on certain Qtools modules so in either case Qtools
must be installed in the local environment.

Using the module
----------------
The script implements the following methods:

	* `Jiang et al. (2015)`_.
	* `Der Kiureghian et al. (1981)`_.
	
The module estimates the ISRS for a 1-DoF system attached to a N-DoF system.
The module assumes that the third direction (OD = 2) is the vertical direction.
A complete documentation is available as a standalone PDF document
(EN/NU/TECH/TGN/031).

Using DirectS2S as a script
---------------------------
DirectS2S can be executed directly as as script. The following instructions
apply:

	* Save this module and the `directs2s` folder in the working directory.
	* Install Qtools somewhere in the local PYTHONPATH.
	* Create and save files `ModalData.txt` and `SpectralData.txt` in the
	  working directiory.
	* Modify the user-defined parameters identified below and save.
	* Execute this module from the command line or through an IDE application.

Calling DirectS2S as a function
-------------------------------
DirectS2S can be called as as function within the Qtools package. For function
documentation, see :func:`directS2S` below. In this approach, the two files
`ModalData.txt` and `SpectralData.txt` are not required. Instead, the input
data are provided directly as arguments `rslist` and `mdlist` to the function.

References
----------
.. _Jiang et al. (2015):
	
	W Jiang, B Li, W-C Xie & MD Pandey, 2015: "Generate floor response spectra,
	Part 1: Direct spectra-to-spectra method", Nuclear Engineering and Design,
	**293**, pp. 525-546.
	
.. _Der Kiureghian et al. (1981):
	
	A Der Kiureghian, JL Sackman & B Nour-Omid, 1981: *Dynamic Response of
	Light Equipment in Structures*, Report No. UBC/EERC-81/05, Earthquake
	Engineering Research Centre, University of California, Berkeley.

"""

import numpy as np
import matplotlib.pyplot as plt
import time
from qtools.response_spectrum import ResponseSpectrum
from qtools import config
from qtools import ds2smods as dsm

if __name__ == '__main__':

	# ------------------------------------------
	#		   USER-DEFINED PARAMETERS
	# ------------------------------------------
	# Method {'Jiang (2015)', 'Der Kiureghian (1981)'}
	method = 'Jiang (2015)'
	
	# Number of excitation directions to be taken into account: {1, 2, 3}
	ND = 3

	# Define output direction of interest: {0, 1, 2}, with 2 being vertical.
	OD = 2

	# Define the desired damping ratio of the ISRS
	z0 = 0.05

	# Plot spectra?
	SPLOT = True

	# Lower bound damping value (for avoidance of division by zero in case a
	# primary mode has been defined with zero damping)
	ZMIN = 0.001

	# Include residual response?
	INRES = True

	# SPECIFIC PARAMETERS REQUIRED FOR 'Jiang (2015)'
	#------------------------------------------------
	# Specify ground type ('H' for hard, 'S' for soft; only relevant for 
	# vertical excitation, odir = 2)
	GT = 'H'

	# Define corner frequencies
	fc = (20,40)

	# ------------------------------------------
	#		END OF USER-DEFINED PARAMETERS
	# ------------------------------------------

def directS2S_main(method, ND, OD, z0, ZMIN, INRES, **kwargs):

	# Retrieve modal information (from textfile if not supplied in **kwargs)
	if 'MD' in kwargs:
		MD = kwargs['MD']
	else:
		MD = np.loadtxt('ModalData.txt')
	fp = MD[:,0]
	zp = np.fmax(MD[:,1],ZMIN)
	gam = MD[:,2:(2+ND)]
	phi = MD[:,(2+ND+OD)]

	# Retrieve response spectra at the base of the primary system.
	# If the SDb parameter is not provided, read spectral input data from file
	# The size of the input file along the 1st axis (i.e. the number of 
	# columns) must be equal to Nzb*ND+1 where Nzb = number of damping values
	# used to define each spectrum.
	if 'SDb' in kwargs:
		SDb = kwargs['SDb']
	else:
		SDb = np.loadtxt('SpectralData.txt')
	Nfb = np.size(SDb,axis=0)-1
	Nzb = (np.size(SDb,axis=1)-1)//ND
	fb = SDb[1:Nfb+1,0]
	zb = SDb[0,1:Nzb+1]
	SAb = np.empty((Nfb,Nzb,ND))
	for d in range(ND):
		SAb[:,:,d] = SDb[1:Nfb+1,d*Nzb+1:(d+1)*Nzb+1]

	# Method Jiang (2015)
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
	else:
		raise ValueError('Method {} is not supported.'.format(method))
	
	rs = ResponseSpectrum(fb, SA_ISRS, xi=z0, label='ISRS {}%'.format(100*z0))
	
	return rs


@config.set_module('qtools')
def directS2S(rslist, mdlist, OD, z0, method='Jiang (2015)', ZMIN=0.001, 
			  INRES=True, **kwargs):
	"""Compute an in-structure response spectrum (ISRS) using a direct
	spectrum-to-spectrum method.

	Parameters
	----------
	rslist : a list of lists of response spectra
		Input spectra for ISRS computation.
		The length of `rslist` determines the number of directions to be taken
		into account (`ND = len(rslist) <= 3`).
		The length of `rslist[0]` determines the number of damping values (`Nzb`).
		If `rslist[1]` is provided, then `len(rslist[1])` must equal `len(rslist[0])`.
		If `rslist[2]` is provided, then `len(rslist[2])` must equal `len(rslist[0])`.
	mdlist : a list of NumPy arrays.
		This parameter contains the modal properties of the primary system,
		where:

		* `mdlist[0]` is a 1D array with natural frequencies (`Np = len(mdlist[0])`);
		* `mdlist[1]` is a 1D array with modal damping ratios;
		* `mdlist[2]` is a 2D array with participation factors
		  (the shape of this array must be `(Np,ND)` or just `(Np,)` if `ND = 1`);
		* `mdlist[3]` is a 2D array with modal displacements (the shape of this
		  array must be `(Np,ND)` or just `(Np,)` if `ND = 1`).

	OD : int, {0, 1, 2}
		Direction of in-structure response.
	z0 : float
		Damping ratio of the secondary system.
	method : str, optional
		The numerical method used to compute the ISRS. Valid options are
		'Jiang (2015)' (default) and 'Der Kiureghian (1981)'.
	ZMIN : float, optional
		Lower bound damping value of the primary system.
		Useful if some primary modes have zero or unrealistically low damping.
		Default 0.001.
	INRES : bool, optional
		Include residual response in the computation. Default `True`.
		See DISRS documentation for further information.

	Other parameters
	----------------
	The following named parameters are used in the Jiang (2015) method:

	fc : tuple of floats, required
		Corner frequencies. See the complete DirectS2S documentation.
	GT : str, optional
		Ground type ('H' for hard, 'S' for soft). Only relevant for vertical
		excitation. Default 'H'.
	
	Returns
	-------
	rs : an instance of class ResponseSpectrum
		The in-structure response spectrum for response in direction `OD`.

	Notes
	-----
	The complete documentation for this function is available as a PDF document:
	EN/NU/TECH/TGN/031, Direct Generation of In-Structure Response Spectra.
	"""
	ND = len(rslist)
	Nzb = len(rslist[0])
	Nf = len(rslist[0][0].f)
	SDb = np.zeros((Nf+1,Nzb+1))
	SDb[1:,0] = rslist[0][0].f
	for i in range(ND):
		for j in range(Nzb):
			SDb[0,ND*i+j+1] = rslist[i][j].xi
			SDb[1:,ND*i+j+1] = rslist[i][j].sa
	Np = len(mdlist[0])
	MD = np.empty((Np,2*ND+2))
	MD[:,0] = np.array(mdlist[0])
	MD[:,1] = np.array(mdlist[1])
	MD[:,2:ND+2] = np.array(mdlist[2]).reshape((Np,ND))
	MD[:,ND+2:2*ND+2] = np.array(mdlist[3]).reshape((Np,ND))
	
	rs = directS2S_main(method, ND, OD, z0, ZMIN, INRES, 
					 SDb=SDb, MD=MD, **kwargs)
	
	return rs

if __name__ == '__main__':
	
	localtime = time.asctime(time.localtime(time.time()))
	print('Start of DirectS2S execution: ',localtime)
	rs = directS2S_main(method, ND, OD, z0, ZMIN, INRES, fc=fc, GT=GT)
	np.savetxt('DirectS2S_Output_{}.txt'.format(OD), np.array([rs.f,rs.sa]).T)
	localtime = time.asctime(time.localtime(time.time()))
	print('End of DirectS2S execution: ',localtime)

	# Plot the in-structure spectra
	if SPLOT:
		# Initialise figure
		plt.figure()
		plt.grid(b=True)
		plt.semilogx(rs.f,rs.sa,label='ISRS')
		plt.legend()
		plt.ylim(bottom=0.0)
		plt.xlim((0.1,100))
		plt.show()