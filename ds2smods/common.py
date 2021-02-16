# -*- coding: utf-8 -*-
"""
Common functions used by two or more methods in DirectS2S.
"""

import numpy as np
from scipy import interpolate
from math import log

def SAfun(SAi, fi, zi, f, z):
	"""
	Interpolation function for spectral accelerations.
	
	Parameters
	----------
	SAi : NumPy array
		A 2D array with spectal accelerations.
	fi : NumPy array
		A 1D array with frequency values. Note: the length of `fi` must equal 
		the length of `SAi` along the first axis.
	zi : NumPy array
		A 1D array with damping values. Note: the length of `zi` must equal 
		the length of `SAi` along the second axis.
	f : NumPy array
		A 1D array with new frequencies.
	z : NumPy Array or float
		New damping value(s). This can be a 1D NumPy array with the same
		length as 'f' or a float.

	Returns
	-------
	SA : NumPy array
		A 1D array of the same length as `f` with spectral accelerations at
		the damping value(s) specified in `z`.
	
	To-do
	-----
	Implement damping interpolation in accordance with ASCE 4-16, 6.2.4.
	"""
	# Check lengths of arguments
	Nf = np.size(f)
	Nz = np.size(z)
	if Nz != Nf and Nz != 1:
		raise IndexError('ERROR: in function SAfun: the input arrays f and z'
				   ' are not the right size.')
	
	# Interpolate along frequency axis (axis 0)
	SAzi = interpolate.interp1d(np.log10(fi), SAi, axis=0, bounds_error=False, 
							fill_value='extrapolate', assume_sorted=True)(np.log10(f))

	# Interpolate along damping axis (axis 1)
	SA = np.empty(Nf)
	for i in range(Nf):
		if Nz == 1:
			SA[i] = np.interp(log(z), np.log(zi), SAzi[i])
		else:
			SA[i] = np.interp(log(z[i]), np.log(zi), SAzi[i])
	
	return SA