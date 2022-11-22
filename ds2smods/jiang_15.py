# -*- coding: utf-8 -*-
"""
This module contains functions needed to implement the Jiang et al (2015)
method.
"""

import numpy as np
from math import pi, exp, log, log10, cos, sqrt
from .common import SAfun
from qtools.config import Info
from numba import jit

def j15_main(SAb, fb, zb, fp, zp, gam, phi, z0, OD, ND, INRES, fc, GT):
	"""
	Main function for the direct spectrum-to-spectrum method developed by
	Jiang et al. (2015).

	Parameters
	----------
	SAb : 3D NumPy array
		Spectral accelerations at the base of the primary structure
		(the input spectrum), with ``SAb[i,j,d]`` returning the spectral
		acceleration at frequency ``fb[i]`` and damping ratio ``zb[j]`` in
		direction `d`.
	fb : 1D NumPy array
		Frequencies corresponding to the 1st axis in `SAb`.
	zb : 1D NumPy array
		Damping ratios corresponding to the 2nd axis in `SAb`.
	fp : 1D NumPy array
		Modal frequencies of the primary system.
	zp : 1D NumPy array
		Damping ratios of the primary system.
	gam : 2D NumPy array
		Participation factors, with ``gam[i,d]`` returning the participation
		factor for mode `i` in direction `d`.
	phi : 1D NumPy array
		Modal displacements for DOF `k` (the point where the secondary
		system is attached).
	z0 : float
		Damping ratio of secondary system.
	OD : int, {0, 1, 2}
		Output direction.
	ND : int, {1, 2, 3}
		Number of directions to take into account.
	INRES : bool
		If True, include the residual response. Otherwise ignore it.
	fc : tuple of 2 floats
		Corner frequencies. See the DirectS2S Documentation.
	GT : str
		Ground type ('H' for hard, 'S' for soft). Only relevant for vertical
		excitation. Default 'H'.

	Returns
	-------
	1D NumPy array
		The in-structure spectral accelerations at the frequencies defined in
		`fb`.
	"""

	Nfb = len(fb)

	# Initialise in-structure spectral acceleration array
	SA_ISRS = np.zeros_like(SAb[:,0,0])

	# Loop over directions
	for d in range(ND):
		Info.note('Computing ISRS for excitation in direction {}'.format(d))
		# Compute spectral accelerations at structural frequencies and damping ratios
		SAp = SAfun(SAb[:,:,d], fb, zb, fp, zp)
		# Compute spectral acceleration at the frequency of the secondary system
		# (note: the effort is unnecessary when z0 equals one of the predefined
		# damping ratios in zb, but for general purpose, we need to to do this)
		SA0 = SAfun(SAb[:,:,d], fb, zb, fb, z0)
		# Compute peak displacements due to excitation in direction d
		u = gam[:,d]*phi
		# Add residual response
		if INRES and d == OD:
			# Append the residual modal displacement to the u array
			ur = 1 - np.sum(u)
			if ur < 0 or ur >= 1:
				Info.warn('With ur = {}, the residual response is outside '
				  'the expected range (0 <= ur < 1)'.format(ur))
				Info.warn('The residual response is ignored (ur = 0)',
				  prepend=False)
				ur = 0
			else:
				Info.note('The residual response is ur = {}'.format(ur))
			u_r = np.append(u,ur)
			# Append appropriate values to other arrays
			# (note: the values appended to fp and zp are somewhat arbitrary)
			SAp_r = np.append(SAp,SAp[-1])
			fp_r = np.append(fp,200)
			zp_r = np.append(zp,z0)

		# Loop over frequency range:
		for i in range(Nfb):
			if INRES and d == OD:
				# Compute the response vector
				R = resp(SAp_r, SA0[i], u_r, fb[i], fp_r, z0, zp_r, fc, d, GT, INRES)
				# Compute the double sum for direction d and add the result
				DS = doublesum(R, fp_r, fb[i], zp_r, z0)
			else:
				# Compute the response vector
				R = resp(SAp, SA0[i], u, fb[i], fp, z0, zp, fc, d, GT, INRES)
				# Compute the double sum for direction d and add the result
				DS = doublesum(R, fp, fb[i], zp, z0)
			SA_ISRS[i] += DS

	# Complete the SRSS combination and return the ISRS
	return np.sqrt(SA_ISRS)

def AR(f, fc, z, d, GT):
	"""
	Evaluate the spectral amplification ratio AR = tRS/GRS.

	Parameters
	----------
	f : float
		Frequency at which the AR should be evaluated.
	fc : array_like
		Corner frequencies (see DirectsS2S documentation).
	z : float
		Damping ratio.
	d : int, {0, 1, 2}
		Direction (with 2 being vertical).
	GT : str, {'H', 'S'}
		Ground type (only used for d=2).
	ZMIN : float
		Minimum damping ratio.

	Returns
	-------
	AR : float
		Spectral amplification ratio.

	Notes
	-----
	This function is partially based on `Li et al. (2015)`_. For more
	information, see the DirectS2S Documentation.

	References
	----------
	.. _Li et al. (2015):

		Li, B; Jiang, W; Xie W-C; & Pandey, M.D., 2015: "Generate floor response
		spectra, Part 2: Response spectra for equipment-structure resonance",
		*Nuclear Engineering and Design*, **293**, pp. 547-560
		(http://dx.doi.org/10.1016/j.nucengdes.2015.05.033).

	"""

	lz = log(100*z)
	if d == 0 or d == 1:
		SAm = 0.02*lz**2 - 0.28*lz + 1.14
		c1 = 0.06*lz**2 - 0.92*lz + 3.03
		c2 = 0.02*lz**3 - 0.04*lz**2 - 0.02*lz + 1.12
	elif d == 2:
		if GT == 'H':
			SAm = 0.03*lz**2 - 0.36*lz + 1.19
			c1 = 0.04*lz**2 - 0.89*lz + 3.09
			c2 = 0.01*lz**4 - 0.06*lz**3 + 0.12*lz**2 - 0.12*lz + 1.15
		elif GT == 'S':
			SAm = 0.04*lz**2 - 0.38*lz + 1.24
			c1 = 0.04*lz**2 - 0.90*lz + 3.13
			c2 = 0.01*lz**3 - 0.03*lz**2 - 0.04*lz + 1.17
		else:
			raise ValueError('GT = {} is not supported'.format(GT))
	f1 = fc[0]
	f2 = fc[1]
	SAt = exp(c1+c2*log(SAm))
	AR1 = SAt/SAm
	if f <= f1:
		AR = AR1
	elif f > f1 and f <= f2:
		AR = (AR1-1)/2*cos(pi*log10(f/f1)/log10(f2/f1))+(AR1+1)/2
	else:
		AR = 1

	return AR

def resp(SAp, SA0, u, f0, fp, z0, zp, fc, d, GT, INRES):
	"""
	Compute the modal response values.

	Parameters
	----------
	SAp : 1D NumPy array
		Spectral accelerations at the frequencies and damping ratios of the
		primary system.
	SA0 : float
		Spectral acceleration at the frequency and damping ratio of the
		secondary system.
	u : 1D NumPy array
		Contribution factors. Must have same length as `SAp`.
	f0 : float
		Frequency of the secondary system.
	fp : 1D NumPy array
		Frequencies of the primary system. Must have same length as `SAp`.
	z0 : float
		Damping ratio of secondary system.
	zp : 1D NumPy array
		Damping ratios of primary system. Must have same length as `SAp`.
	fc : array_like
		Corner frequencies (passed to :func:`AR`).
	d : int, {0, 1, 2}
		Direction (passed to :func:`AR`).
	GT : str, {'H', 'S'}
		Ground type (only used for d=2).
	INRES : boolean
		If `True`, include residual response.

	Returns
	-------
	R : 1D NumPy array
		The modal response values.
	"""

	N = len(SAp)
	R = np.empty_like(SAp)
	AR0 = AR(f0, fc, z0, d, GT)
	for i in range(N):
		if INRES and i == N-1:
			# Residual response
			# (TO-DO: this may be superfluous as AFi --> 0 for r --> inf)
			R[i] = u[i]*SA0
		else:
			# Normal response
			ri = fp[i]/f0
			zi = zp[i]
			ARi = AR(fp[i], fc, zi, d, GT)
			zie = sqrt(2)*zi/(z0+zi)/ARi
			z0e = sqrt(2)*z0/(z0+zi)/AR0
			AF0i = ri**2/sqrt((1-ri**2)**2 + (2*z0e*ri)**2)
			AFi = 1/sqrt((1-ri**2)**2 + (2*zie*ri)**2)
			R[i] = u[i]*sqrt((AF0i*SA0)**2 + (AFi*SAp[i])**2)

	return R

@jit(nopython=True)
def doublesum(R, f, f0, z, z0):
	"""
	Compute the double sum.

	Parameters
	----------
	R : 1D NumPy array
		Modal response values.
	f : 1D NumPy array
		Modal frequencies. Must have same length as `R`.
	f0 : float
		Frequency of the secondary system.
	z : 1D NumPy array
		Modal damping ratios. Must have same length as `R`.
	z0 : float
		Damping ratio of secondary system.

	Returns
	-------
	SS : float
		The double sum.

	"""
	SS = 0
	N = len(R)
	for i in range(N):
		ri = f[i]/f0
		zi = z[i]
		Ri = R[i]
		SS += Ri**2
		rj = f[i+1:]/f0
		zj = z[i+1:]
		Rj = R[i+1:]
		# Compute correlation coefficient
		Dij1 = 1 - 2*rj**2 + rj**4 + 4*z0*zj*rj + 4*z0*zj*rj**3 + 4*z0**2*rj**2 + 4*zj**2*rj**2
		Dij2 = 1 - 2*ri**2 + ri**4 + 4*z0*zi*ri + 4*z0*zi*ri**3 + 4*z0**2*ri**2 + 4*zi**2*ri**2
		Dij3 = (ri**2-rj**2)**2 + 4*zi*zj*ri*rj*(ri**2+rj**2) + 4*ri**2*rj**2*(zi**2+zj**2)
		Cij0 = (1 - ri**2 - rj**2 + ri**2*rj**2 + 4*zi*zj*ri*rj)*Dij3
		Cij1 = 4*(2*zi*ri + 2*zj*rj + 8*zi*zj*ri*rj*(zi*ri+zj*rj) - 4*(zi*ri**3+zj*rj**3)
				  + 8*zi**3*ri**3 + 8*zj**3*rj**3 - 2*ri*rj*(zi*rj**3+zj*ri**3)
				  + 8*zi*zj*ri*rj*(zi*ri**3+zj*rj**3) + 4*ri**2*rj**2*(zi*ri+zj*rj)
				  - 8*ri**2*rj**2*(zi**3*ri+zj**3*rj) - 8*zi*zj*ri**2*rj**2*(zi*rj+zj*ri)
				  + 32*zi**2*zj**2*ri**2*rj**2*(zi*ri+zj*rj) + ri*rj*(zi*rj**5+zj*ri**5)
				  + ri**2*rj**2*(zi*ri**3+zj*rj**3) + 4*zi*zj*ri**2*rj**2*(zi*rj**3+zj*ri**3)
				  - 2*ri**3*rj**3*(zi*rj+zj*ri) + 4*ri**3*rj**3*(zi**3*rj+zj**3*ri)
				  + 8*zi*zj*ri**3*rj**3*(zi*ri+zj*rj))
		Cij2 = 4*(8*zi**2*ri**2 + 8*zj**2*rj**2 + 16*zi*zj*ri*rj + 64*zi**2*zj**2*ri**2*rj**2
				  - 4*zi*zj*ri*rj*(ri**2+rj**2) + 32*zi*zj*ri*rj*(zi**2*ri**2+zj**2*rj**2)
				  + 6*ri**2*rj**2 - 12*ri**2*rj**2*(zi**2+zj**2) - 3*(ri**4+rj**4)
				  + 8*zi*zj*ri*rj*(ri**4+rj**4) - ri**2*rj**2*(ri**2+rj**2) + 8*zi**2*ri**4
				  + 8*zj**2*rj**4 + 4*ri**2*rj**2*(zi**2+zj**2)*(ri**2+rj**2)
				  + 16*zi**2*zj**2*ri**2*rj**2*(ri**2+rj**2) + 16*zi*zj*ri**3*rj**3*(zi**2+zj**2)
				  + ri**6 + rj**6)
		Cij3 = 16*(8*zi*zj*ri*rj*(zi*ri+zj*rj) + 2*zi*ri**3 + 2*zj*rj**3 + ri*rj*(zi*rj**3+zj*ri**3)
				   + 4*zi*zj*ri*rj*(zi*ri**3+zj*rj**3) - 2*ri**2*rj**2*(zi*ri+zj*rj)
				   + 4*ri**2*rj**2*(zi**3*ri+zj**3*rj) + 8*zi*zj*ri**2*rj**2*(zi*rj+zj*ri)
				   + zi*ri**5 + zj*rj**5)
		Cij4 = 16*Dij3
		aij = 1/(Dij1*Dij2*Dij3)*(Cij0+Cij1*z0+Cij2*z0**2+Cij3*z0**3+Cij4*z0**4)
		bi = ((z0+4*z0**2*zi*ri+4*z0*zi**2*ri**2+zi*ri**3)/
			  (zi*ri**3*(1-2*ri**2+ri**4+4*z0*zi*ri+4*(zi**2+z0**2)*ri**2+4*z0*zi*ri**3)))
		bj = ((z0+4*z0**2*zj*rj+4*z0*zj**2*rj**2+zj*rj**3)/
			  (zj*rj**3*(1-2*rj**2+rj**4+4*z0*zj*rj+4*(zj**2+z0**2)*rj**2+4*z0*zj*rj**3)))
		rhoij = aij/np.sqrt(bi*bj)
		SS += 2*Ri*np.sum(rhoij*Rj)

	return SS