"""
Module qtools.power_spectrum
Contains class PowerSpectrum
Author: Andreas H Nielsen
"""

import numpy as np
import sys
from scipy.integrate import romb
from math import pi, sqrt, ceil

class PowerSpectrum:

	""" This class provides methods for calculation of Fourier and power spectra.
	
	Usage:
	
	>>> ps = PowerSpectrum(method,ordinate,**kwargs)	

	Method is one of the following:
   
	method = 'transient' (default):
	Calculate a spectrum from a time history.
	Required parameters:
	th = an instance of class TimeHistory.
	Optional parameters:
	None.
	
	ordinate is one of the following:

	ordinate = 'Fourier amplitude'
	The ordinates of the spectrum are the Fourier amplitudes calculated by means of FFT

	ordinate = 'Power amplitude' 
	The ordinates of the spectrum are Fourier amplitudes squared
		
	**kwargs is a list of parameters as required by each method/output.
	
	Common options:
   
   

	"""
	
	def __init__(self,method='transient',ordinate='Power amplitude',**kwargs):
	
		method = method.lower()
		self.label = '_nolegend_'

		if method=='transient':
			try:
				th = kwargs.pop('th')
			except KeyError:
				print('Error: in class PowerSpectrum: parameter th not specified with method transient')
				sys.exit(1)
			
			if not th.dt_fixed:
				print('Error: in class PowerSpectrum: a time history must be defined with fixed intervals')
				sys.exit(1)
			
			# Ensure time history has zero mean
			th.zero_mean()
	
			# Pad with zeros
			N = th.ndat
			p = 1			 
			while N-2**p > 0:
				p += 1
			M = 2**p
			a = np.zeros(M)
			a[:N] = th.data
			print('Size of original time history = ',N)
			print('Size of zero-padded time history = ',M)
			# Execute FFT
			Xc = 1/M*np.fft.rfft(a) 
			K = np.size(Xc)
			# Fourier amplitude			   
			self.X = np.absolute(Xc)
			TL = th.dt*M
			T = th.time[-1]
			print('Duration of original time history =',T)
			self.Sk = M/N*TL/(2*pi)*self.X**2
			self.Sw = np.zeros(K)
			# Assume the required bandwidth is 0.5Hz
			Be = 0.5
			ns = ceil(M*Be*T/(2*N)-0.5)
			print('Smoothing parameter ns = ',ns)
			# Calculate smoothed spectrum
			for k in range(K):
				sswkm = 0.0
				for m in range(-ns,ns+1):
					i = k+m
					if i < 0:
						sswkm += self.Sk[-i]
					elif i >= 0 and i < K-1:
						sswkm += self.Sk[i]
				self.Sw[k] = 1/(2*ns+1)*sswkm
			self.Wf = 4*pi*self.Sw
			self.f = np.fft.rfftfreq(M,d=th.dt)
			
			# Label
			self.label = 'PS of '+th.label
			
	def rms(self):
		
		"""
		This function calculates the root mean square from the power spectrum.
		The function assumes that the spectrum contains 2**(p-1)+1 values
		equi-spaced along the frequency axis and with self.f[1] equal to the frequency spacing.
		It also assumes that the spectrum is symmetric about f = 0.
		"""
		df = self.f[1]
		IWf = romb(self.Wf,dx=df)
		rms = sqrt(IWf)
		
		return rms
			
	def export(self,fname,abscissa='f', ordinate='Sw', fmt='%.18e',
			   delimiter=' ', newline='\n', header='', footer='', comments='# '):
			  
		if abscissa=='f' and ordinate=='Sk':	
			out = np.array([self.f,self.Sk])
   
		elif abscissa=='T' and ordinate=='Sk':	  
			out = np.array([self.T,self.Sk])

		elif abscissa=='f' and ordinate=='Sw':	  
			out = np.array([self.f,self.Sw])

		elif abscissa=='T' and ordinate=='Sw':	  
			out = np.array([self.T,self.Sw])

		elif abscissa=='f' and ordinate=='X':	 
			out = np.array([self.f,self.Xc])

		elif abscissa=='T' and ordinate=='X':	 
			out = np.array([self.T,self.X])

		elif abscissa=='f' and ordinate=='all':	   
			out = np.array([self.f,self.X,self.Sk,self.Sw])

		elif abscissa=='T' and ordinate=='all':	   
			out = np.array([self.T,self.X,self.Sk,self.Sw])
			
		np.savetxt(fname,out.T,fmt=fmt,delimiter=delimiter,newline=newline,
					   header=header,footer=footer,comments=comments)