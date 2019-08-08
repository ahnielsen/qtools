'''
Direct generation of in-structure response spectra (ISRS).
The script implements the method proposed by Jiang et al. (2015 paper).
The script estimates the ISRS for a 1-DoF system attached to a N-DoF system.
The script assumes that the third direction (du = 2) is the vertical direction
'''

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from math import sqrt, log10, log, exp, cos, pi
import time
from qtools.response_spectrum import ResponseSpectrum

if __name__ == '__main__':
	
	# ------------------------------------------
	#		   USER DEFINED PARAMETERS
	# ------------------------------------------
	# Number of excitation directions to be taken into account (D = 1, 2 or 3)
	D = 3
	
	# Define output direction of interest (du = 0, 1, or 2)
	du = 2
	
	# Define the desired damping ratio of the ISRS
	z0 = 0.05
	
	# Plot spectra?
	SPLOT = True
	
	# Specify ground type ('H' for hard, 'S' for soft; only relevant for vertical excitation)
	GT = 'H'
	# Define corner frequencies
	fc = (20,40)
	
	# Lower bound damping value (for avoidance of division by zero in case of zero damping)
	ZMIN = 0.001
	
	# Include residual response?
	INRES = True
	#INRES = False
	
	# ------------------------------------------
	#		END OF USER DEFINED PARAMETERS
	# ------------------------------------------

def disrs_main(D,du,z0,fc,GT,ZMIN,INRES,**kwargs):

	# Import modal information
	if 'MD' in kwargs:
		MD = kwargs['MD']
	else:
		MD = np.loadtxt('ModalData.txt')
	Np = np.size(MD,axis=0)
	fp = MD[:,0]
	zp = MD[:,1]
	gam = MD[:,2:(2+D)]
	phi = MD[:,(2+D+du)]
	
	# If the SDb parameter is not provided, read spectral input data from file
	# The size of the input file along the 1st axis (i.e. the number of columns) must be equal to Nzb*D+1
	# where Nzb = number of damping values used to define each spectrum
	if 'SDb' in kwargs:
		SDb = kwargs['SDb']
	else:
		SDb = np.loadtxt('SpectralData.txt')
	Nfb = np.size(SDb,axis=0)-1
	Nzb = (np.size(SDb,axis=1)-1)//D
	fb = SDb[1:Nfb+1,0]
	zb = SDb[0,1:Nzb+1]
	SAb = np.empty((Nfb,Nzb,D))
	for d in range(D):
		SAb[:,:,d] = SDb[1:Nfb+1,d*Nzb+1:(d+1)*Nzb+1]
	
	# Create interpolation functions for spectral accelerations
	def SAfun(SAi,fi,zi,f,z):
		# Impose a lower bound on the damping value
		z = np.fmax(z,ZMIN*np.ones_like(z))
		Nf = np.size(f)
		Nz = np.size(z)
		Nzi = np.size(zi)
		if Nz != Nf and Nz != 1:
			raise IndexError('ERROR: in function SAfun: the input arrays f and z are not the same size')
		SAiz = interpolate.interp1d(np.log10(fi),SAi,axis=0,bounds_error=True,assume_sorted=True)(np.log10(f))
		SAo = np.empty(Nf)
		for i in range(Nf):
			if Nzi > 1:
				SAofun = interpolate.interp1d(np.log10(zi),SAiz[i],axis=0,bounds_error=False,fill_value="extrapolate",assume_sorted=True)
				if Nz == 1:
					SAo[i] = SAofun(log10(z))
				else:
					SAo[i] = SAofun(log10(z[i]))
			else:
				SAo[i] = SAiz[i]
		return SAo
	
	# Define the spectral amplification ratio AR = tRS/GRS
	def AR(f,fc,z,d):
		# Impose a lower bound on the damping value
		z = max(z,ZMIN)
		try:
			lz = log(100*z)
		except ValueError:
			raise ValueError('In function AR with z = {}'.format(z))
		if d == 0 or d == 1:
			SAm = 0.02*lz**2 - 0.25*lz + 1.14
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
	
	# Define function to compute the double sum
	def doublesum(SA,SA0,u,N,f0,fp,fc,z0,zp,d,inres):
		SS = 0
		AR0 = AR(f0,fc,z0,d)
		for i in range(N):
			if inres and i == N-1:
				# Residual response
				Ri = u[i]*SA0
			else:
				# Normal response
				ri = fp[i]/f0
				zi = max(zp[i],ZMIN)
				ARi = AR(fp[i],fc,zi,d)
				zie = sqrt(2)*zi/(z0+zi)/ARi
				z0e = sqrt(2)*z0/(z0+zi)/AR0
				AF0i = ri**2/sqrt((1-ri**2)**2+(2*z0e*ri)**2)
				AFi = 1/sqrt((1-ri**2)**2+(2*zie*ri)**2)
				Ri = u[i]*sqrt((AF0i*SA0)**2+(AFi*SA[i])**2)
			SS += Ri**2
			for j in range(i+1,N):
				if inres and j == N-1:
					# Residual response
					rj = fp[j]/f0
					zj = max(zp[j],ZMIN)
					Rj = u[j]*SA0
				else:
					# Normal response
					rj = fp[j]/f0
					zj = max(zp[j],ZMIN)
					ARj = AR(fp[j],fc,zj,d)
					zje = sqrt(2)*zj/(z0+zj)/ARj
					AF0j = rj**2/sqrt((1-rj**2)**2+(2*z0e*rj)**2)
					AFj = 1/sqrt((1-rj**2)**2+(2*zje*rj)**2)
					Rj = u[j]*sqrt((AF0j*SA0)**2+(AFj*SA[j])**2)
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
				rhoij = aij/sqrt(bi*bj)
				SS += 2*rhoij*Ri*Rj
	
		return SS
	
	# Initialise in-structure spectral acceleration array
	SA_ISRS = np.zeros_like(SAb[:,0,0])
	
	# Loop over directions
	for d in range(D):
		print('Direction {}'.format(d))
		# Compute spectral accelerations at structural frequencies and damping ratios
		SAp = SAfun(SAb[:,:,d],fb,zb,fp,zp)
		# Compute spectral acceleration at the frequency of the secondary system
		# (note: the effort is unnecessary when z0 equals one of the predefined damping ratios in SpectralData.txt, but for general purpose, we need to to do this)
		SA0 = SAfun(SAb[:,:,d],fb,zb,fb,z0)
		# Compute peak displacements due to excitation in direction d
		u = gam[:,d]*phi
		# Add residual response
		if INRES and d == du:
			# Append the residual modal displacement to the u array
			ur = 1 - np.sum(u)
			if ur < 0 or ur >= 1:
				print('WARNING: with ur = {}, the redidual response is outside the expected range (0 <= ur < 1)'.format(ur))
				print('The residual response is ignored (ur = 0)')
				ur = 0
			else:
				print('The residual response is ur = {}'.format(ur))
			u_r = np.append(u,ur)
			# Append appropriate values to other arrays (note: the values appended to fp and zp are somewhat arbitrary)
			SAp_r = np.append(SAp,SAp[-1])
			fp_r = np.append(fp,200)
			zp_r = np.append(zp,z0)
	
		# Loop over frequency range:
		for i in range(Nfb):
			# Compute the double sum for direction d and add the result
			if INRES and d == du:
				DS = doublesum(SAp_r,SA0[i],u_r,Np+1,fb[i],fp_r,fc,z0,zp_r,d,True)
			else:
				DS = doublesum(SAp,SA0[i],u,Np,fb[i],fp,fc,z0,zp,d,False)
			SA_ISRS[i] += DS
	
	# Complete the SRSS combination and generate response spectrum
	SA_ISRS = np.sqrt(SA_ISRS)
	rs = ResponseSpectrum(fb,SA_ISRS,xi=z0)
	rs.setLabel('ISRS {}%'.format(100*z0))
	return rs

if __name__ == '__main__':
	
	localtime = time.asctime(time.localtime(time.time()))
	print('Start of DISRS execution: ',localtime)
	rs = disrs_main(D,du,z0,fc,GT,ZMIN,INRES)
	np.savetxt('DISRS_Output_{}.txt'.format(du),np.array([rs.f,rs.sa]).T)
	localtime = time.asctime(time.localtime(time.time()))
	print('End of execution: ',localtime)
	
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

def disrs(rslist,mdlist,du,z0,fc,GT='H',ZMIN=0.001,INRES=True):
	'''Compute an in-structure response spectrum (ISRS) using a direct spectrum-to-spectrum method.
	
	Parameters
	----------
	rslist : a list of lists of response spectra
		Input spectra for ISRS computation.
		The length of rslist determine the number of directions to be taken into account (D <= 3).
		The length of rslist[0] determines the number of damping values (Nzb).
		If rslist[1] is provided, then len(rslist[1]) must equal len(rslist[0]).
		If rslist[2] is provided, then len(rslist[2]) must equal len(rslist[0]).
	mdlist : a list of Numpy arrays.
		Where:
			
			* mdlist[0] is a 1D array with natural frequencies (Np = len(mdlist[0]));
			* mdlist[1] is a 1D array with damping ratios;
			* mdlist[2] is a 2D array with participation factors (the shape of this array must be (Np,D) or just (Np,) if D = 1);
			* mdlist[3] is a 2D array with modal displacements (the shape of this array must be (Np,D) or just (Np,) if D = 1).
	
	du : int
		Direction of in-structure response. ``[0|1|2]``
	z0 : float
		Damping value of the secondary system.
	fc : tuple of floats
		Corner frequencies. See DISRS documentation.
	GT : str
		Ground type ('H' for hard, 'S' for soft). Only relevant for vertical excitation.
	ZMIN : float
		Lower bound damping value of the primary system.
		Useful if some primary modes have zero or unrealistically low damping.
		Default 0.001.
	INRES : boolean
		Include residual response in the computation. Default ``True``.
		See DISRS documentation for further information.
	
	Returns
	-------
	rs : an instance of class ResponseSpectrum
		The in-structure response spectrum for response in direction ``du``.
	'''
	D = len(rslist)
	Nzb = len(rslist[0])
	Nf = len(rslist[0][0].f)
	SDb = np.zeros((Nf+1,Nzb+1))
	SDb[1:,0] = rslist[0][0].f
	for i in range(D):
		for j in range(Nzb):
			SDb[0,D*i+j+1] = rslist[i][j].xi
			SDb[1:,D*i+j+1] = rslist[i][j].sa
	Np = len(mdlist[0])
	MD = np.empty((Np,2*D+2))
	MD[:,0] = np.array(mdlist[0])
	MD[:,1] = np.array(mdlist[1])
	MD[:,2:D+2] = np.array(mdlist[2]).reshape((Np,D))
	MD[:,D+2:2*D+2] = np.array(mdlist[3]).reshape((Np,D))
	rs = disrs_main(D,du,z0,fc,GT,ZMIN,INRES,SDb=SDb,MD=MD)
	return rs
