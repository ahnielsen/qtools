# -*- coding: utf-8 -*-
"""
This module contains functions needed to implement the Der Kiureghian et al
(1981) method.
"""

import numpy as np
from math import sqrt
from .common import SAfun
from qtools import config
from numba import jit

#@jit(nopython=True)
def dk81_main(SAb, fb, zb, fp, zp, gam, phi, me, ze, OD, ND, INRES, 
			 Mi=None):
	"""
	Main function for the direct spectrum-to-spectrum method developed by Der
	Kiureghian et al. (1981).
	
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
	me : float
		Mass of secondary system.
	ze : float
		Damping ratio of secondary system.
	OD : int
		Output direction.
	ND : int
		Number of directions to take into account.
	INRES : bool
		If True, include the residual response. Otherwise ignore it.
	Mi : 1D NumPy array, optional
		Modal masses. The default is None, in which case it is assumed that
		all modes have been normalised to the mass matrix (in other words, it
		is assumed that ``Mi = np.ones_like(phi)``).

	Returns
	-------
	1D NumPy array
		The in-structure spectral accelerations at the frequencies defined in
		`fb`.
	"""
	
	with open('debug.txt', 'w') as outfil:
		
		# If Mi is None, then all modes have been normalised such that Mi = 1
		if Mi is None:
			Mi = np.ones_like(phi)
		
		# Determine lengths of input arrays
		Nfb = len(fb)
		Np = len(fp)
		
		# Initialise arrays
		pkis = np.empty(Np+1)
		pnis = np.empty(Np+1)
		fps = np.empty(Np+1)
		zps = np.empty(Np+1)
		Mis = np.empty(Np+1)
		Gis = np.empty(Np+1)
		Rt = np.empty(Np+1)
		SA_ISRS = np.zeros(Nfb)
	
		# Loop over directions
		for d in range(ND):
			config.vprint('Computing ISRS for excitation in direction {}'.format(d))
			# DEBUG start
			outfil.write('Direction {}\n'.format(d))
			# DEBUG end
			# Loop over frequency range:
			for i in range(Nfb):
				# Parameters
				bi = (fp**2 - fb[i]**2)/fb[i]**2
				gi = me*phi**2/Mi
				redInd = np.argwhere(np.isclose(bi,0) & np.isclose(gi,0))
				if len(redInd) > 0:
					gi[redInd] = 1e-3
					for i in redInd[:,0]:
						config.vprint('WARNING: increased equipment mass in mode '
					   '{} to avoid numerical instability. New mass value: {}'
					   .format(i,gi[i]*Mi[i]/phi[i]**2))
				A1 = 1 + (bi+gi)/2 - np.sqrt((1 + (bi+gi)/2)**2 - (1+bi))
				A2 = 1 + (bi+gi)/2 + np.sqrt((1 + (bi+gi)/2)**2 - (1+bi))
				denom = np.where(bi < 0, A1-1, A2-1)
				ai = -1/denom
				sag = np.sum(ai*gi)
				sa2g = np.sum(ai**2*gi)
				ci = np.sqrt(1+bi)
				l = np.argmin(np.fabs(bi))
				rn = 1 if d == OD else 0
				# New modal frequencies
				fps[0] = sqrt(1 + sag)*fb[i]
				fps[1:] = fp*np.where(bi < 0, np.sqrt(A1/(1+bi)), np.sqrt(A2/(1+bi)))
				# New modal damping ratios
				zps[0] = (np.sum(ci*ai**2*gi*zp) + (1+sag)**2*ze)/(1+sa2g)
				zps[1:] = (ci*zp + (1-ai)**2*gi*ze)/(ci*(1+ai**2*gi))
				# New modal displacement (DOF k)
				pkis[0] = -sag
				pkis[1:] = phi
				pkis[l+1] = sag - ai[l]*gi[l] - 1/ai[l]
				# New modal displacement (DOF n+1)
				pnis[0] = 1
				pnis[1:] = ai*pkis[1:]
				pnis[l+1] = -1
				# New modal masses
				Mis[0] = (1+sa2g)*me
				Mis[1:] = (1+ai**2*gi)*Mi
				a2gl = ai[l]**2*gi[l]
				Mis[l+1] = (1 + a2gl*(1+sa2g-a2gl))*Mi[l]/(ai[l]*phi[l])**2
				# New participation factors
				Gis[0] = -(np.sum(ai*phi*gam[:,d])-rn)/(1+sa2g)
				Gis[1:] = (gam[:,d]+ai*gi*rn/phi)/(1+ai**2*gi)
				Gis[l+1] = -((ai[l]*phi[l]*gam[l,d]-ai[l]**2*gi[l]*
					(np.sum(ai*phi*gam[:,d])-ai[l]*phi[l]*gam[l,d]-rn))/
					(1+ai[l]*gi[l]*(1+sa2g-a2gl)))
				# Compute spectral accelerations at new frequencies
				SAp = SAfun(SAb[:,:,d], fb, zb, fps, zps)
				# Modal responses
				Rt = Gis*(pnis-pkis)*SAp/fps**2
				# Compute the double sum for direction d and add the result
				DS = doublesum(Rt, fps, zps)
				SA_ISRS[i] += fb[i]**4*DS

	# Complete the SRSS combination and return the ISRS
	return np.sqrt(SA_ISRS)

@jit(nopython=True)
def doublesum(R, f, z):
	"""
	Compute the double sum.

	Parameters
	----------
	R : 1D NumPy array
		Modal response values.
	f : 1D NumPy array
		Modal frequencies. Must have same length as `R`.
	z : 1D NumPy array
		Modal damping ratios. Must have same length as `R`.

	Returns
	-------
	SS : float
		The double sum.

	"""
	SS = 0
	for i in range(len(R)):
		Ri = R[i]
		Rj = R[i+1:]
		fi = f[i]
		fj = f[i+1:]
		zi = z[i]
		zj = z[i+1:]
		rhoij = (2*np.sqrt(zi*zj)*((fi+fj)**2*(zi+zj)+(fi**2-fj**2)*(zi-zj))/
		        (4*(fi-fj)**2+(zi+zj)**2*(fi+fj)**2))
		SS += Ri**2
		SS += 2*Ri*np.sum(rhoij*Rj)

	return SS