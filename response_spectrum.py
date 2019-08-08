"""
Module qtools.response_spectrum
Author: Andreas H Nielsen
"""

import numpy as np
import copy
import functools
import multiprocessing
from scipy.integrate import odeint
from scipy import interpolate
from math import exp, log, log10, pi, sqrt, floor, sin, cos, isclose
from qtools.systems import vis1, sld
from qtools import config
from pathlib import Path

class ResponseSpectrum:
	"""
	General class for response spectra.
	
	Parameters
	----------
	x : 1D Numpy array
		Frequency or period values.
	y : 1D Numpy array of same length as x
		Spectral ordinates (acceleration, velocity or displacement)
	abscissa : string, optional, ``['f'|'T']``
		Specify quantity contained in x, with 'f' = frequency and 'T' = period. Default 'f'.
	ordinate : string, optional, ``['sag'|'sa'|'sv'|'sd']
		Specify quantity contained in y with:
		
		* 'sag' = acceleration (units of g)
		* 'sa' = acceleration (units of m/s**2)
		* 'sv' = velocity (units of m/s)
		* 'sd' = displacement (units of m)

		Default 'sag'.
	xi : float, optional
		Damping ratio. Default 0.05.
	
	Attributes
	----------
	sa : Numpy array
		Spectral acceleration in units of g.
	sv : Numpy array
		Spectral velocity in units of m/s.
	sd : Numpy array
		Spectral displacement in units of m
	f : Numpy array
		Frequency in units of Hz (1/s)
	T : Numpy array
		Period in units of s (= 1/f)
	g : float
		Value of the acceleration due to gravity 
	xi : float
		Damping ratio
	ndat : integer
		Number of elements in sa, sv, sd, T, and f.
	
	A note on units
	---------------
	* Each instance of ResponseSpectrum is defined in terms of standard units.
	* The standard spatial dimension is metres.
	* The standard time dimension is seconds.
	* The standard unit for spectral acceleration is g (with g = 9.81 m/s**2).

	Special methods
	---------------
	In the following, rs1 and rs2 are instances of ResponseSpectrum.
	str(rs1) : calls __str__(self)
		Returns a string representation of response spectrum rs1;  this will also work with the built-in functions ``format(rs1)`` and ``print(rs1)``.
	rs1 + rs2 : calls __add__(self,other)
		Adds two response spectra (note: this has yet to be implemented).
	a*rs1 : calls __rmul__(self,other)
		Multiplies the spectral ordinates of rs1 with a where a must be an integer or a float.
	
	Instance methods
	----------------
	rs1.setLabel(s) : with argument s (string)
		Sets the label of rs1 equal to s.
	rs1.export() : with no arguments
		Exports rs1 to a text file (see function for arguments).
	rs1.interp(f,option='retain',eps=0.001) : for arugments, see below. 
		Interpolates the spectral ordinates of rs1 at frequencies contained in Numpy array f.
		Arguments:
		
			* option = 'retain': retain the orginal frequencies of rs1 (rs1.f and f are combined).
			* option = 'replace': replace the original frequencies of rs1.
			* eps: precision; if option = 'retain', duplicate frequencies defined as frequencies that satisfy log(f2/f1) < eps are combined into one average frequency.
	"""
	
	def __init__(self,x,y,abscissa='f',ordinate='sag',xi=0.05):

		self.g = 9.81

		if abscissa=='f':
			self.f = x
			self.T = self._invert(x)
		elif abscissa=='T':
			x = np.flip(x)
			y = np.flip(y)
			self.T = x
			self.f = self._invert(x)
		else:
			raise ValueError('In class ResponseSpectrum, the abscissa has to be frequency (f) or period (T)')

		omega = 2*pi*self.f
		if ordinate == 'sag':
			self.sa = y
			self.sv = self.g*self.sa/omega
			self.sd = self.sv/omega
		elif ordinate == 'sa':
			self.sa = y/self.g
			self.sv = self.g*self.sa/omega
			self.sd = self.sv/omega
		elif ordinate == 'sv':
			self.sv = y
			self.sd = self.sv/omega
			self.sa = omega*self.sv/self.g
		elif ordinate == 'sd':
			self.sd = y
			self.sv = omega*self.sd
			self.sa = omega*self.sv/self.g
		
		self.xi = xi
		self.ndat = len(y)
		self.k = 0
		self.label = '_no label_'
			
	def __iter__(self):
		return self

	def __next__(self):
		if self.k < self.ndat:
			f = self.f[self.k]
			sa = self.sa[self.k]
			self.k += 1
		else:
			self.k = 0
			raise StopIteration
		return (f,sa)
	
	def __str__(self):
		s = self.label+'\nNumber of points: '+str(self.nf)
		return s
		
	def __add__(self,other):
		if type(other) != ResponseSpectrum:
			print('Error: you cannot add a response spectrum to an object of a different type.')
			return self
		else:
			print('Warning: addition of two response spectra has yet to be implemented.')
			return self
	
	def __rmul__(self,other):
		if (type(other) != float or type(other) != int):
			print('Error: you can only multiply a response spectrum with a float or an integer.')
			return self
		else:
			product = copy.deepcopy(self)
			product.sa *= other
			product.sv *= other
			product.sd *= other
		return product
		
	def _invert(self,a):
		b = np.empty_like(a)
		if a[0]==0.0:
			b[0] = 10**(-floor(log10(a[1])))
			b[1:] = 1/a[1:]
		else:
			b = 1/a
		return b
		
	def setLabel(self,label):
		self.label = str(label)
		
	def savers(self,fname,abscissa='f',ordinate='sag',fmt='%.18e',
			   delimiter=' ',newline='\n',header='',footer='',comments='# '):
		
		if header=='':
			head1 = 'Response spectrum computed by Qtools v. {}\n'.format(config.version)
			head2 = 'Damping level = {}'.format(self.xi)
			header = head1+head2
		
		if abscissa=='f' and ordinate=='sag':	
			out = np.array([self.f,self.sa])

		elif abscissa=='T' and ordinate=='sag':	  
			out = np.array([self.T[::-1],self.sa[::-1]])

		elif abscissa=='f' and ordinate=='sv':	  
			out = np.array([self.f,self.sv])

		elif abscissa=='T' and ordinate=='sv':	  
			out = np.array([self.T[::-1],self.sv[::-1]])

		elif abscissa=='f' and ordinate=='sd':	  
			out = np.array([self.f,self.sd])

		elif abscissa=='T' and ordinate=='sd':	  
			out = np.array([self.T[::-1],self.sd[::-1]])

		elif abscissa=='f' and ordinate=='all':	   
			out = np.array([self.f,self.sa,self.sv,self.sd])
			header += '\nFreq, SA, SV, SD'

		elif abscissa=='T' and ordinate=='all':
			out = np.array([self.T[::-1],self.sa[::-1],self.sv[::-1],self.sd[::-1]])
			
		np.savetxt(fname,out.T,fmt=fmt,newline=newline,delimiter=delimiter,
					   header=header,footer=footer,comments=comments)
		
	def interp(self,fn,merge=True,eps=0.001,sd_extrapolate=True):
		'''
		Interpolate a spectrum at the frequencies given as the first argument to the function.
		
		Parameters
		----------
		fn : 1D Numpy array
			New frequencies (the spectrum will be interpolated at these frequencies)
		merge : boolean, optional
			Set ``merge=True`` to merge the new frequencies with the existing frequencies of the spectrum.
			Set ``merge=False`` to replace the existing frequencies with the new frequencies.
		eps : float, optional
			If frequencies are merged, duplicate frequencies defined as frequencies that satisfy log10(f2/f1)) < eps
			are combined into one average frequency. The default value of eps is 0.001.
		sd_extrapolate :: boolean, optional
			Manner of extrapolation for f < self.f[0] (see implementation notes below). Default ``True``.
		
		Implementation notes
		--------------------
		Spectral accelerations are interpolated on a logarithmic frequency scale.
		The method will extrapolate for f < self.f[0] and for f > self.f[-1].
		It is assumed that the spectral acceleration is constant for f > self.f[-1].
		If ``sd_extrapolate`` is ``True``, then it is assumed that the spectral displacement is constant for f < self.f[0].
		If ``sd_extrapolate`` is ``False``, then it is assumed that the spectral acceleration is constant for f < self.f[0].
		''' 
		if merge:
			fn = np.sort(np.append(self.f,fn))
			# Remove duplicate frequencies within a precision of eps (on a log scale)
			ft = np.zeros_like(fn)
			ft[0] = fn[0]
			j = 0
			for i in range(1,np.size(fn)):
				if log10(fn[i]/ft[j]) < eps:
					ft[j] = sqrt(fn[i]*ft[j])
				else:
					ft[j+1] = fn[i]
					j += 1
			fn = ft[0:j+1]
		isa = interpolate.interp1d(np.log10(self.f),self.sa,bounds_error=False,fill_value=(self.sa[0],self.sa[-1]),assume_sorted=True)
		sa = isa(np.log10(fn))
		if sd_extrapolate:
			for i in range(np.size(fn)):
				if fn[i] < self.f[0]:
					sa[i] = self.sd[0]*(2*pi*fn[i])**2/self.g
				else:
					break
		self.f = fn
		self.T = self._invert(self.f)
		omega = 2*pi*self.f
		self.sa = sa
		self.sv = self.g*self.sa/omega
		self.sd = self.sv/omega
		self.ndat = len(sa)
		self.k = 0

def envelope(rs1,rs2,option=0,sd_extrapolate=True):
	'''
	Compute the envelope of two spectra.
	
	Parameters
	----------
	rs1 : an instance of ResponseSpectrum
		First spectrum for the enveloping operation.
	rs2 : an instance of ResponseSpectrum
		Second spectrum for the enveloping operation.
	option : integer, optional
		Set ``option=0`` to envelope at the frequencies of both spectra and any frequencies
		where the two spectra intersect (recommended).
		Set ``option=1`` or ``option=2`` to envelope only at the frequencies of the 1st or 2nd spectrum and any frequencies
		where the two spectra intersect	(note: this may result in imperfect enveloping).
	sd_extrapolate : boolean, optional
		Manner of extrapolation for f < fmin where fmin is the lowest frequency used to defined rs1 or rs2 (see implementation notes below). Default ``True``.
	
	Returns
	-------
	rs : an instance of ResponseSpectrum
		The envelope of rs1 and rs2
		
	Implementation notes
	--------------------
	Spectral accelerations are interpolated and/or extrapolated as necessary
	to compute the envelope over the entire frequency range. The ResponseSpectrum method
	``interp()`` is used for this purpose. The parameter ``sd_extrapolate`` is passed
	directly to ``interp()``. That is, when extrapolation is performed,
	
		* it is assumed that the spectral acceleration is constant for f > fmax;
		* when ``sd_extrapolate``, it is assumed that the spectral displacement is constant for f < fmin;
		* when ``not sd_extrapolate``, it is assumed that the spectral acceleration is constant for f < fmin.
	
	'''
	# Create copies of rs1 and rs1 to avoid changing the values of their attributes
	rsa = copy.deepcopy(rs1)
	rsb = copy.deepcopy(rs2)
	# Define a function that determines whether two lines intersect
	def intersection(xa,ya,xb,yb):
		coords = (0,0)
		x1 = log10(xa[0]); x2 = log10(xa[1])
		y1 = ya[0]; y2 = ya[1]
		x3 = log10(xb[0]); x4 = log10(xb[1])
		y3 = yb[0]; y4 = yb[1]
		denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
		# Lines are parallel if denom == 0
		if not isclose(denom,0):
			t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom
			u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denom
			if (t >= 0 and t <= 1) and (u >= 0 and u <= 1):
				xi = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
				yi = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
				coords = (10**xi,yi)
		return coords	
	# Find intersections
	k = 0
	fi = []
	for i in range(len(rsa.f)-1):
		for j in range(k,len(rsb.f)-1):
			if rsb.f[j] >= rsa.f[i+1]: break
			if rsb.f[j+1] <= rsa.f[i]: k += 1; continue
			coords = intersection(rsa.f[i:i+2],rsa.sa[i:i+2],rsb.f[j:j+2],rsb.sa[j:j+2])
			if coords[0] > 0:
				fi.append(coords[0])
	# Interpolate spectra at the specified frequencies
	if option == 0:
		rsa.interp(np.unique(np.append(rsb.f,fi)),sd_extrapolate=sd_extrapolate)
		rsb.interp(rsa.f,sd_extrapolate=sd_extrapolate)
		f0 = rsa.f
	elif option == 1:
		rsb.interp(np.unique(np.append(rsa.f,fi)),merge=False,sd_extrapolate=sd_extrapolate)
		f0 = rsa.f
	elif option == 2:
		rsa.interp(np.unique(np.append(rsb.f,fi)),merge=False,sd_extrapolate=sd_extrapolate)
		f0 = rsb.f
	else:
		raise ValueError('ERROR: in function envelope(): option must be assigned a value of 0, 1 or 2.')
	# Envelope
	sa0 = np.fmax(rsa.sa,rsb.sa)
	# Remove redundant points (points on a straight line)
	f1 = np.array([])
	sa1 = np.array([])
	a0 = (sa0[1]-sa0[0])/(log10(f0[1]/f0[0]))+1
	for i in range(len(f0)-1):
		a1 = (sa0[i+1]-sa0[i])/(log10(f0[i+1]/f0[i]))
		if not isclose(a1,a0):
			a0 = a1
			f1 = np.append(f1,f0[i])
			sa1 = np.append(sa1,sa0[i])
	f1 = np.append(f1,f0[-1])
	sa1 = np.append(sa1,sa0[-1])
	# Finally, check damping
	if rs1.xi != rs2.xi:
		print('WARNING: an envelope is created of two spectra with different damping ratios.')
		print('The damping ratio of the new enveloping spectrum is defined as {}.'.format(rs1.xi))
	
	return ResponseSpectrum(f1,sa1,xi=rs1.xi)

def peakbroad(rs,df=0.15,df_sign='plus/minus',peak_cap=True):
	''' Broaden a spectrum.
	
	Parameters
	----------
	rs : an instance of class ResponseSpectrum
		Input spectrum for broadening operation.
	df : float, optional
		The relative bandwidth for broadening. The default value (``df=0.15``) corresponds
		to 15% peak broadening. The value assigned to df must be positive.
		A value ``df=0`` results in no broadening.
	df_sign : string, optional, ``['plus'|'minus'|'plus/minus']``
		The sign implied on parameter ``df``. If for example ``df_sign = 'minus'``, the function
		will undertake peak broadening in the negative direction only.
	peak_cap : boolean
		Cap the peaks before the broadening operation in accordance with ASCE 4-16, 6.2.3(b).
	'''
	#
	# Note: need to implement optional truncation of broadening at rs.f[0] and rs.f[-1]
	#
	# Split the spectrum into segments
	sgn0 = np.sign(rs.sa[1]-rs.sa[0])
	fseg = []; saseg = []
	rsl = []; sgn = []
	for i in range(rs.ndat):
		fseg.append(rs.f[i])
		saseg.append(rs.sa[i])
		if i < rs.ndat-1:
			sgn1 = np.sign(rs.sa[i+1]-rs.sa[i])
			if sgn1 == sgn0:
				# Segment continues
				continue
			else:
				# Sign change - start a new segment
				rsl.append([np.array(fseg),np.array(saseg)])
				sgn.append(sgn0)
				fseg = [rs.f[i]]
				saseg = [rs.sa[i]]
				sgn0 = sgn1
		else:
			# Finalise last segment
			rsl.append([np.array(fseg),np.array(saseg)])
			sgn.append(sgn0)
	
	# Check that we have at least 2 segments
	if len(sgn) < 2:
		raise ValueError('Cannot peak-broaden a spectrum with {} segments.'.format(len(sgn)))

	# Cap the peaks (if required) and fill the valleys
	if peak_cap:
		i = 0
		while i <= len(sgn)-2:
			if sgn[i] == 1 and sgn[i+1] == -1:
				rsl_sub, sgn_sub = _peak_cap(rsl[i],rsl[i+1])
				rsl = rsl[:i]+rsl_sub+rsl[i+2:]
				sgn = sgn[:i]+sgn_sub+sgn[i+2:]
			i += 1
	
	# Re-assemble the spectrum, filling in the valleys
	f0 = rsl[0][0][0]
	sa0 = rsl[0][1][0]
	i = 0
	while i <= len(sgn)-2:
		if sgn[i] == -1 and sgn[i+1] == 1:
			if df_sign == 'plus':
				val = _valley_fill(rsl[i],rsl[i+1],df=df,rtf='l')
			elif df_sign == 'minus':
				val = _valley_fill(rsl[i],rsl[i+1],df=df,rtf='r')
			else:
				val = _valley_fill(rsl[i],rsl[i+1],df=df,rtf='r')
			f0 = np.append(f0,val[0][1:])
			sa0 = np.append(sa0,val[1][1:])
			i += 2
		else:
			f0 = np.append(f0,rsl[i][0][1:])
			sa0 = np.append(sa0,rsl[i][1][1:])
			i += 1
	if i == len(sgn)-1:
		f0 = np.append(f0,rsl[i][0][1:])
		sa0 = np.append(sa0,rsl[i][1][1:])
	
	# Peak-broaden the re-assembled spectrum 
	# Initialise values
	if df_sign == 'plus':
		f1, sa1 = _broaden(f0,sa0,df,0)
		rsb = ResponseSpectrum(f1,sa1,xi=rs.xi)
	elif df_sign == 'minus':
		f1, sa1 = _broaden(f0,sa0,0,df)
		rsb = ResponseSpectrum(f1,sa1,xi=rs.xi)
	elif df_sign == 'plus/minus':
		f1p, sa1p = _broaden(f0,sa0,df,0)
		f1m, sa1m = _broaden(f0,sa0,0,df)
		rsb = envelope(ResponseSpectrum(f1p,sa1p,xi=rs.xi),ResponseSpectrum(f1m,sa1m,xi=rs.xi))
	else:
		print('WARNING: df_sign = {} is not a valid option in function peakbroad.'.format(df_sign))
		rsb = rs
	
	return rsb

def _valley_fill(left,right,df=0.15,rtf='c'):
	'''Auxiliary function used by function peakbroad()
	Fill a valley at df relative bandwidth (default 15%). Takes as input two arrays, left and right, with:
		
		* left[0] = an increasing frequency array (left[0][i] < left[0][i+1])
		* left[1] = a descending acceleration array (left[1][i] > left[1][i+1])
		* right[0] = an increasing frequency array (right[0][i] < right[0][i+1])
		* right[1] = an increasing acceleration array (right[0][i] < right[0][i+1])
	
	It is assumed that left[0][-1] = right[0][0] and left[1][-1] = right[1][0]
	-  i.e. the two segments share the lowest point.
	
	The third parameter, ``direction``, specifies how the 15% is calculated:
		
		* rtf = 'l': relative to the frequencies on the left side of the valley;
		* rtf = 'c': relative to the central frequency (fc = left[0][-1] = right[0][0]);
		* rtf = 'r': relative to the frequencies on the right side of the valley;
	'''
	
	# Segments
	fL0 = left[0]; saL0 = left[1]
	fL = np.flip(fL0); saL = np.flip(saL0)
	fR = right[0]; saR = right[1]
	# Central frequency
	fc = fR[0]
	# Minimum start and end point acceleration
	sap_min = min(saL[-1],saR[-1])
	# Interpolate/extrapolate frequencies (enrich left and right segments with additional points)
	ip_fL = interpolate.interp1d(saL,np.log(fL),bounds_error=False,fill_value='extrapolate',assume_sorted=True)
	ip_fR = interpolate.interp1d(saR,np.log(fR),bounds_error=False,fill_value='extrapolate',assume_sorted=True)
	sae = np.unique(np.append(saL,saR))
	fLe = np.exp(ip_fL(sae))
	fRe = np.exp(ip_fR(sae))
	if rtf == 'l':
		dfe = (fRe-fLe)/fLe
	elif rtf == 'r':
		dfe = (fRe-fLe)/fRe
	else:
		dfe = (fRe-fLe)/fc
	# Find the spectral acceleration with a valley relative bandwidth of df (if it exists)
	sa1 = sap_min
	for i in range(len(dfe)-1):
		if dfe[i+1] >= df:
			aR = (fRe[i+1]-fRe[i])/(sae[i+1]-sae[i])
			bR = fRe[i]-aR*sae[i]
			aL = (fLe[i+1]-fLe[i])/(sae[i+1]-sae[i])
			bL = fLe[i]-aL*sae[i]
			if rtf == 'l':
				sa1 = -(df*bL-bR+bL)/(df*aL-aR+aL)
			elif rtf == 'r':
				sa1 = -(df*bR-bR+bL)/(df*aR-aR+aL)
			else:
				sa1 = (df*fc-bR+bL)/(aR-aL)
			break
	sa1 = min(sa1,sap_min)
	fL1 = np.exp(ip_fL(sa1))
	fR1 = np.exp(ip_fR(sa1))
	# Return a new segment
	fseg = np.concatenate((fL0[saL0 > sa1],np.array([fL1,fR1]),fR[saR > sa1]))
	saseg = np.concatenate((saL0[saL0 > sa1],np.array([sa1,sa1]),saR[saR > sa1]))
	return [fseg,saseg]

def _peak_cap(left,right):
	'''Auxiliary function used by function peakbroad()
	Cap a peak in accordance with ASCE 4-16, 6.2.3(b). Takes as input two arrays, left and right, with:
		
		* left[0] = an increasing frequency array (left[0][i] < left[0][i+1])
		* left[1] = a increasing acceleration array (left[1][i] < left[1][i+1])
		* right[0] = an increasing frequency array (right[0][i] < right[0][i+1])
		* right[1] = an descending acceleration array (right[0][i] > right[0][i+1])
	
	It is assumed that left[0][-1] = right[0][0] and left[1][-1] = right[1][0]
	-  i.e. the two segments share the peak point. 
	'''
	
	# Segments
	fL = left[0]; saL = left[1]
	fR0 = right[0]; saR0 = right[1]
	fR = np.flip(fR0); saR = np.flip(saR0)
	# Central frequency
	fc = fL[-1]
	# Maximum start and end point acceleration
	sa0_max = max(saL[0],saR[0])
	# Interpolate frequencies
	ip_fL = interpolate.interp1d(saL,np.log(fL),assume_sorted=True)
	ip_fR = interpolate.interp1d(saR,np.log(fR),assume_sorted=True)
	# Determine whether peak capping should be applied
	peak_cap = False
	if sa0_max < 0.8*saL[-1]:
		fL80 = exp(ip_fL(0.8*saL[-1]))
		fR80 = exp(ip_fR(0.8*saL[-1]))
		df80 = (fR80-fL80)/fc
		if df80 < 0.3:
			sa85 = 0.85*saL[-1]
			fL85 = exp(ip_fL(sa85))
			fR85 = exp(ip_fR(sa85))
			peak_cap = True
	# Return new segments
	if peak_cap:
		rsls = [[np.append(fL[saL < sa85],fL85),np.append(saL[saL < sa85],sa85)],
		   [np.array([fL85,fR85]),np.array([sa85,sa85])],
		   [np.append(fR0[saR0 < sa85],fR85),np.append(saR0[saR0 < sa85],sa85)]]
		sgns = [1,0,-1]
	else:
		rsls = [left,right]
		sgns = [1,-1]
	return rsls,sgns
	
def _broaden(f0,sa0,dfp,dfm):
	''' Auxiliary function used by function peakbroad()
	'''
	sgn0 = np.sign(sa0[1]-sa0[0])
	if sgn0 < 0:
		f1 = (1+dfp)*f0[0]
	elif sgn0 > 0:
		f1 = (1-dfm)*f0[0]
	else:
		f1 = f0[0]
	sa1 = sa0[0]
	for i in range(len(f0)-1):
		sgn1 = np.sign(sa0[i+1]-sa0[i])
		if sgn0 == 1 and sgn1 == 1:
			# Case (1)
			f1 = np.append(f1,(1-dfm)*f0[i+1])
			sa1 = np.append(sa1,sa0[i+1])
		elif sgn0 == 1 and sgn1 == -1:
			# Case (2)
			f1 = np.append(f1,[(1+dfp)*f0[i],(1+dfp)*f0[i+1]])
			sa1 = np.append(sa1,[sa0[i],sa0[i+1]])
		elif sgn1 == 0:
			# Case (3), (6) or (9)
			f1 = np.append(f1,(1+dfp)*f0[i+1])
			sa1 = np.append(sa1,sa0[i+1])
		elif sgn0 == 0 and sgn1 == 1:
			# Case (4)
			if isclose(f1[-2],(1-dfm)*f0[i]):
				# The sequence (2)+(9)+(4) can lead to a coincident point which we don't want to retain.
				j = -2
			else:
				j = -1 
			f1 = np.append(f1[0:j],[(1-dfm)*f0[i],(1-dfm)*f0[i+1]])
			sa1 = np.append(sa1[0:j],[sa0[i],sa0[i+1]])
		elif sgn0 == 0 and sgn1 == -1:
			# Case (5)
			f1 = np.append(f1,(1+dfp)*f0[i+1])
			sa1 = np.append(sa1,sa0[i+1])
		elif sgn0 == -1 and sgn1 == 1:
			# Case (7)
			print('WARNING: A case (7) has been found. Check results carefully.')
			continue
		elif sgn0 == -1 and sgn1 == -1:
			# Case (8)
			f1 = np.append(f1,(1+dfp)*f0[i+1])
			sa1 = np.append(sa1,sa0[i+1])
		sgn0 = sgn1
	return (f1,sa1)
	
def calcrs(th,ffile=None,nf=200,fmin=0.1,fmax=100,xi=0.05,solver='solode',accuracy='medium',MP=False):
	'''
	Calculate a response spectrum from a time history.
	
	Parameters
	----------
	th : an instance of class TimeHistory
		An acceleration time history defining the base motion.
	ffile : string, optional 
		Read frequency points from file ``ffile``. Default value ``None``.
		If ``ffile=None`` the function will use ``nf``, ``fmin`` and ``fmax`` to generate an array of frequency points.
	nf : integer, optional
		Number of frequency points. Default 200.
	fmin : float, optional
		Minimum frequency considered. Default 0.1 Hz.
	fmax : float, optional
		Maximum frequency considered. Default 100.0 Hz.
		The function will generate ``nf`` frequency points between ``fmin`` and ``fmax`` equally spaced on a logarithmic scale.
	xi : float, optional
		Damping ratio. Default 0.05.
	solver : string, optional, ``['solode'|'odeint'|'ode']``
		Solution algorithm to be use for the computation. Default 'solode'.
	accuracy : string, optional, ``['low'|'medium'|'high']``
		The required accuracy of the response spectrum. Valid with solver odeint only.
		The specified accuracy can affect high frequency values.
	MP : boolean, optional
		Use multi-processing to compute the response spectrum. Default ``False``.
		Multi-processing is only implemented for the solode solver.
	
	Returns
	-------
	**rs :** an instance of ResponseSpectrum 
	
	Solvers
	-------
	The default solver is solode, which is particularly suited to a second order linear (SOL) ordinary differential equation (ODE).
	The odeint and ode solvers are from SciPy's collection of ODE solvers. These solvers are more general in application.
	The odeint solver is recommended for verification purposes. The ode solver is retained for historical reasons.
	'''

	# Code involving dml is legacy code which can be ignored. 
	dml = 'viscous'
	
	# Preliminary checks
	if th.ordinate != 'a':
		raise TypeError('The time history used as input to function calcrs must be an acceleration time history.')
	if xi >= 1.0 and solver=='solode':
		raise ValueError('The damping ratio must be less than 1 when using the solode solver.')
	if solver=='odeint':
		config.vprint('Using solver odeint to compute response spectrum with accuracy = {}.'.format(accuracy))
		if dml != 'viscous':
			config.vprint('NOTE: The damping model is: {}'.format(dml))
	elif solver=='solode':
		config.vprint('Using solver solode to compute response spectrum.')
		if th.dt_fixed:
			config.vprint('The time step is taken as fixed.')
		else:
			config.vprint('The time step is taken as variable.')
			
	# Generate frequency points
	if ffile==None:
		# Generate frequency points
		f = np.logspace(log10(fmin),log10(fmax),nf)
	else:
		# Read frequency points from file
		f = np.loadtxt(ffile)
		nf = np.size(f)
		fmax = f[-1]
	if f[0] >= th.fNyq:
		raise ValueError('The lowest frequency of the spectrum is greater than, or equal to, the Nyquist frequency of the time history.')

	omega = 2*pi*f[f<=th.fNyq]
	y0 = [0.0,0.0]
	if solver == 'odeint':
		sd = np.empty(len(f))
		for i in range(len(omega)):
			if dml == 'viscous':
				fun1 = vis1
			elif dml == 'solid':
				fun1 = sld
			if accuracy == 'low':
				# Low accuracy - for quick and dirty solutions
				sol = odeint(fun1,y0,th.time,args=(omega[i],xi,th),atol=0.0001,rtol=0.001)
			elif accuracy == 'medium': 
				# The atol and rtol values in the following call seem to provide a reasonable compromise between speed and accuracy
				sol = odeint(fun1,y0,th.time,args=(omega[i],xi,th),atol=0.000001,rtol=0.001)
			elif accuracy == 'high':
				# For solutions that display spurious high frequency behaviour, the following call should be tried:
				sol = odeint(fun1,y0,th.time,args=(omega[i],xi,th))
			sd[i] = np.amax(np.fabs(sol[:,0]))
			# The following five lines calculate the total energy at the last time step as the sum of kinetic energy, strain energy and energy dissipated in viscuos damping.
			# Ekin = 1/2*sol[-1,1]**2 # Note: this assumes that the input velocity is zero at the last time step
			# Estr = 1/2*(omega*sol[-1,0])**2
			# intg = 2*xi*omega*sol[:,1]**2
			# Evis = trapz(intg,x=th.time)
			# wk[i] = Evis+Ekin+Estr
			# -----------------------
			# The following three lines estimates the work done on the system. This should yield similar results.
			# svel = cumtrapz(th.data,th.time,initial=0)
			# intg = -(2*xi*omega*sol[:,1]+omega**2*sol[:,0])*svel
			# Wext = trapz(intg,x=th.time)
			# -----------------------
	elif solver == 'solode':
		processes = max(multiprocessing.cpu_count()-1, 1)
		if processes > 1 and MP:
			# Multiple processes
			config.vprint('Will use {} processes to compute spectrum'.format(processes))
			config.vprint('Remember to protect the main module with an if __name__ == \'__main__\': statement')
			with multiprocessing.Pool(processes=processes) as pool:
				sd = pool.map(functools.partial(_solode, y0, th, xi), omega)
		else:
			# Single process
			sd = np.array([_solode(y0,th,xi,w) for w in omega])
			
	# Set spectral accelerations equal to the PGA for f greater than the Nyquist frequency
	if fmax > th.fNyq:
		config.vprint('NOTE: Spectral accelerations above the Nyquist frequency ({:6.2f} Hz) are assumed equal to the ZPA.'.format(th.fNyq))
		sd = np.where(f <= th.fNyq, sd, th.pga/(2*pi*f)**2)
		
	# Complete the creation of a response spectrum
	config.vprint('------------------------------------------')
	rs = ResponseSpectrum(f,sd,abscissa='f',ordinate='sd',xi=xi)
	rs.setLabel(th.label)
	return rs

def _solode(y0,th,xi,w,peak_resp_only=True):
	'''
	Solver for a second order linear ordinary differential equation.
	This solver is intended for a single degree-of-freedom system subject to base excitation.
	
	Parameters
	----------
	y0 : tuple or list of floats
		Initial conditions: y0[0] = initial displacement, y0[1] = initial velocity
	th : instance of class TimeHistory
		Time history defining the base acceleration
	xi : float
		Damping ratio of the system
	w : float
		Undamped natural frequency of the system

	Returns
	-------
	If ``peak_resp_only == True``:
	y : float
		The maximum absolute displacement (spectral displacement)

	If ``peak_resp_only == False``:
	y : Numpy array, shape (len(th.ndat),2)
		Array containing the value of y for each time point in th.time, with the initial value y0 in the first row.

	'''

	wd = w*sqrt(1-xi**2)
	a = np.empty((2,2))
	b = np.empty((2,2))
	n = th.ndat
	y = np.empty((n,2))
	y[0,:] = y0
	f = th.data
	if th.dt_fixed:
		dt = th.dt
		c = cos(wd*dt)
		s = sin(wd*dt)
		e = exp(-xi*w*dt)
		a[0,0] = e*(xi*w*s/wd+c)
		a[0,1] = e*s/wd
		a[1,0] = -e*w*s/sqrt(1-xi**2)
		a[1,1] = e*(c-xi*w*s/wd)
		b[0,0] = -e*((xi/(w*wd)+(2*xi**2-1)/(w**2*wd*dt))*s+(1/w**2+2*xi/(w**3*dt))*c)+2*xi/(w**3*dt)
		b[0,1] = e*((2*xi**2-1)/(w**2*wd*dt)*s+2*xi/(w**3*dt)*c)+1/w**2-2*xi/(w**3*dt)
		b[1,0] = -e*((wd*c-xi*w*s)*((2*xi**2-1)/(w**2*wd*dt)+xi/(wd*w))-(wd*s+xi*w*c)*(1/w**2+2*xi/(w**3*dt)))-1/(w**2*dt)
		b[1,1] = -e*(-(wd*c-xi*w*s)*(2*xi**2-1)/(wd*w**2*dt)+(wd*s+xi*w*c)*2*xi/(w**3*dt))+1/(w**2*dt)
		for i in range(n-1):
			y[i+1,:] = a@y[i,:]+b@[f[i],f[i+1]]
	else:
		t = th.time
		dt0 = 0
		for i in range(n-1):
			dt = t[i+1]-t[i]
			# Ignore very small variations in time step
			if not isclose(dt,dt0):
				c = cos(wd*dt)
				s = sin(wd*dt)
				e = exp(-xi*w*dt)
				a[0,0] = e*(xi*w*s/wd+c)
				a[0,1] = e*s/wd
				a[1,0] = -e*w*s/sqrt(1-xi**2)
				a[1,1] = e*(c-xi*w*s/wd)
				b[0,0] = -e*((xi/(w*wd)+(2*xi**2-1)/(w**2*wd*dt))*s+(1/w**2+2*xi/(w**3*dt))*c)+2*xi/(w**3*dt)
				b[0,1] = e*((2*xi**2-1)/(w**2*wd*dt)*s+2*xi/(w**3*dt)*c)+1/w**2-2*xi/(w**3*dt)
				b[1,0] = -e*((wd*c-xi*w*s)*((2*xi**2-1)/(w**2*wd*dt)+xi/(wd*w))-(wd*s+xi*w*c)*(1/w**2+2*xi/(w**3*dt)))-1/(w**2*dt)
				b[1,1] = -e*(-(wd*c-xi*w*s)*(2*xi**2-1)/(wd*w**2*dt)+(wd*s+xi*w*c)*2*xi/(w**3*dt))+1/(w**2*dt)
			dt0 = dt
			y[i+1,:] = a@y[i,:]+b@[f[i],f[i+1]]
	
	if peak_resp_only:	
		return np.amax(np.fabs(y[:,0]))
	else:
		return y
				
def loadrs(sfile,abscissa='f',ordinate='sag',length='m',xi=0.05,delimiter=None,comments='#',skiprows=0):
	'''
	Load a response spectrum from a text file. The x-values (abscissae) must be in the first column,
	and the y-values (ordinates) must in the second column of the file.
	Note that the specification of xi in this function has no effect on the ordinates.
	The input spectrum may be in units of L, L/s, L/s**2 (where L is the length dimension, e.g. metres) or in units of g.
	The following are examples of valid specifications:
		* ``ordinate = 'sa'`` and ``length = 'm'``: the ordinates are defined in units of m/s**2.
		* ``ordinate = 'sag'`` and ``length = 'm'``: the ordinates are defined in units of g.
		* ``ordinate = 'sv'`` and ``length = 'cm'``: the ordinates are defined in units of cm/s.
		* ``ordinate = 'sd'`` and ``length = 'mm'``: the ordinates are defined in units of mm/s.

	Parameters
	----------
	sfile : string
		The path and name of the input file.
	abscissa : string, optional, ``['f'|'T']``
		The physical quantity of the data in the first column of the input file with
		'f' = frequency and 'T' = period (default 'f').
	ordinate : string, optional, ``['sa'|'sag'|'sv'|'sd']``
		The physical quantity of the data in the second column of the input file (default 'sa').
		
		* 'sa' = spectral acceleration (unit L/s**2)
		* 'sag' = spectral acceleration (unit g)
		* 'sv' = spectral velocity (unit L/s)
		* 'sd' = spectral displacement (unit L)
	
	length : string, optional, ``['m'|'dm'|'cm'|'mm']``
		Unit of length dimension L.
		
	See ``numpy.loadtxt`` for further information on parameters ``delimiter``, ``comments`` and ``skiprows``.
	'''

	lf = {'m': 1, 'dm': 10, 'cm': 100, 'mm': 1000}
	inp = np.loadtxt(sfile,delimiter=delimiter,comments=comments,skiprows=skiprows)
	x = inp[:,0]
	y = inp[:,1]/lf[length]
	rs = ResponseSpectrum(x,y,abscissa=abscissa,ordinate=ordinate,xi=xi)
	# Set label (removing the file path and the extension, if any)
	rs.setLabel(Path(sfile).stem)
	return rs

def ec8_2004(pga,pga_unit='g',xi=0.05,inf=10,stype=1,gtype='A',q=1.0,option='HE'):
	'''
	Create a response spectrum in accordance with BS EN 1998-1:2004+A1:2013.
	This function will generate a spectrum in the range T = [0.01;10] sec or f = [0.1;100] Hz.
	
	Parameters
	----------
	pga : float
		Peak ground acceleration
	pga_unit : string, optional, ``['g'|'m/s**2']``
		Unit of the pga value. Default value 'g'.
	xi : float, optional
		Damping ratio. Default value 0.05.
	inf : integer, optional
		Number of points in each interval resulting in a total of ``ndat = 4*inf`` points. Default value 10.
	stype : integer, optional, ``[1|2]``
		Specify Type 1 or 2 spectrum. Default value 1.
	gtype : string, optional, ``['A'|'B'|'C'|'D'|'E']``
		Ground type. Default value 'A'.
	q : float, optional
		Behaviour factor.
	option : string, optional, ``['HE'|'HD'|'VE'|'VD']``
		Type of spectrum: *H*orizontal / *V*ertical, *E*lastic / *D*esign.
		The default value, 'HE', is the horizontal elastic spectrum.
	'''

	if gtype not in ('A','B','C','D','E'):
		raise ValueError('Unsupported ground type in call to ec8_2004: gtype = {}'.format(gtype))
	
	if stype == 1:
		params = {'A' : [1.0, 0.15, 0.4, 2.0],
				  'B' : [1.2, 0.15, 0.5, 2.0],
				  'C' : [1.15, 0.20, 0.6, 2.0],
				  'D' : [1.35, 0.20, 0.8, 2.0],
				  'E' : [1.4, 0.15, 0.5, 2.0]}			
	elif stype == 2:
		params = {'A' : [1.0, 0.05, 0.25, 1.2],
				  'B' : [1.35, 0.05, 0.25, 1.2],
				  'C' : [1.5, 0.10, 0.25, 1.2],
				  'D' : [1.8, 0.10, 0.30, 1.2],
				  'E' : [1.6, 0.05, 0.25, 1.2]}
	
	S = params[gtype][0]
	TB = params[gtype][1]
	TC = params[gtype][2]
	TD = params[gtype][3]
	eta = max(sqrt(10./(5+100*xi)),0.55)
	
	Tl = list(np.linspace(0.01,TB,inf,endpoint=False))
	Tl = Tl+list(np.linspace(TB,TC,inf,endpoint=False))
	Tl = Tl+list(np.linspace(TC,TD,inf,endpoint=False))
	Tl = Tl+list(np.linspace(TD,10,inf,endpoint=True)) 
	
	SA = []		 
	if option=='HE':
		for T in Tl:
			if T < TB:
				SA.append(pga*S*(1+T/TB*(eta*2.5-1)))
			elif T < TC:
				SA.append(pga*S*eta*2.5)
			elif T < TD:
				SA.append(pga*S*eta*2.5*TC/T)
			else:
				SA.append(pga*S*eta*2.5*TC*TD/T**2)
	elif option=='HD':
		print('Warning: option HD has not been implemented yet.')
	elif option=='VE':
		print('Warning: option VE has not been implemented yet.')
	elif option=='VD':
		print('Warning: option VD has not been implemented yet.')
	else:
		raise ValueError('Unsupported option in call to ec8_2004: option = {}'.format(option))
	
	if pga_unit == 'g':
		ord = 'sag'
	else:
		ord = 'sa'
	
	rs = ResponseSpectrum(np.array(Tl),np.array(SA),abscissa='T',ordinate=ord,xi=xi)
	rs.setLabel('EC8 Type '+str(stype)+' GT '+str(gtype)+' '+str(xi*100)+'%')
	return rs
	
def dec8_2017(SRP,sa_unit='g',xi=0.05,inf=10,gtype='A',q=1.0,option='HE'):
	'''
	Create a response spectrum in accordance with the draft revision of BS EN 1998-1 (2017).
	This function will generate a spectrum in the range T = [0.01;10] sec or f = [0.1;100] Hz.
	
	Parameters
	----------
	SRP : tuple of floats with 2 elements
		The two spectral parameters S_sRP and S_1RP. 
	sa_unit : string, optional, ``['g'|'m/s**2']``
		Unit of the spectral parameters. Default value 'g'.
	xi : float, optional
		Damping ratio. Default value 0.05.
	inf : integer, optional
		Number of points in each interval resulting in a total of ``ndat = 5*inf`` points. Default value 10.
	gtype : string, optional, ``['A'|'B'|'C'|'D'|'E']``
		Ground type. Default value 'A'.
	q : float, optional
		Behaviour factor.
	option : string, optional, ``['HE'|'HD'|'VE'|'VD']``
		Type of spectrum: *H*orizontal / *V*ertical, *E*lastic / *D*esign.
		The default value, 'HE', is the horizontal elastic spectrum.
	'''
	
	# Define spectral parameters 
	F0 = 2.5
	chi = 4
	Ss = SRP[0]
	S1 = SRP[1]
	print('Ss = {:}, S1 = {:}'.format(Ss,S1))
	T1 = 1.0
	TA = 0.03
	TC = S1*T1/Ss
	
	if TC/chi < 0.05:
		TB =  0.05
	elif TC/chi > 0.1:
		TB = 0.1
	else:
		TB = TC/chi
	if S1 < 0.1:
		TD = 2.0
	else:
		TD = 1.0+10*S1
	print('TA = {:}, TB = {:}, TC = {:}, TD = {:}'.format(TA,TB,TC,TD))
	print('fD = {:}, fC = {:}, fB = {:}, fA = {:}'.format(1/TD,1/TC,1/TB,1/TA))
	
	if gtype not in ('A','B','C','D','E'):
		raise ValueError('Unsupported ground type in call to dec8_2017: gtype = {}'.format(gtype))
	
	eta = max(sqrt(10./(5+100*xi)),0.55)
	
	Tl = list(np.logspace(log10(0.01),log10(TA),num=inf,endpoint=False))
	Tl = Tl+list(np.logspace(log10(TA),log10(TB),num=inf,endpoint=False))
	Tl = Tl+list(np.logspace(log10(TB),log10(TC),num=inf,endpoint=False))
	Tl = Tl+list(np.logspace(log10(TC),log10(TD),num=inf,endpoint=False))
	Tl = Tl+list(np.logspace(log10(TD),log10(10),num=inf,endpoint=True))
	
	SA = []
	if option=='HE':
		for T in Tl:
			if T < TA:
				SA.append(Ss/F0)
			elif T < TB:
				SA.append(Ss/(TB-TA)*(eta*(T-TA)+(TB-T)/F0))
			elif T < TC:
				SA.append(eta*Ss)
			elif T < TD:
				SA.append(eta*S1*T1/T)
			else:
				SA.append(eta*TD*S1*T1/T**2)
	elif option=='HD':
		print('Warning: option HD has not been implemented yet.')
	elif option=='VE':
		print('Warning: option VE has not been implemented yet.')
	elif option=='VD':
		print('Warning: option VD has not been implemented yet.')
	else:
		raise ValueError('Unsupported option in call to dec8_2017: option = {}'.format(option))
	
	if sa_unit == 'g':
		ord = 'sag'
	else:
		ord = 'sa'
	
	rs = ResponseSpectrum(np.array(Tl),np.array(SA),abscissa='T',ordinate=ord,xi=xi)	
	rs.setLabel('DEC8 GT '+str(gtype)+' '+str(xi*100)+'%')
	return rs
	
def ieee693_2005(option='moderate',inf=10,xi=0.02):
	'''
 	Generate the required response spectrum in accordance with IEEE Std 693-2005.
	
	Parameters
	----------
	option : string, optional, ``['moderate'|'high']``
		Specify qualification level. Default 'moderate'.
	inf : integer, optional
		Number of points in each interval resulting in a total of ``nf = 4*inf`` points. Default value 10.
	xi : float, optional
		Damping ratio. Default value 0.02.
	'''
	
	a = (0.572,0.625,6.6,2.64,0.2,0.33,0.25)
	if option == 'high':
		a *= 2.0
	elif option != 'moderate':
		print('Warning: in in function ieee693_2005: could not recognise IEEE qualification level.')
		print('Assuming moderate level.')

	f0 = 0.3; f1 = 1.1; f2 = 8.0; f3 = 33.0; f4 = 50.0
	beta = (3.21-0.68*log(100*xi))/2.1156
	
	fl = list(np.logspace(log10(f0),log10(f1),inf,endpoint=False))
	fl = fl+list(np.logspace(log10(f1),log10(f2),inf,endpoint=False))
	fl = fl+list(np.logspace(log10(f2),log10(f3),inf,endpoint=False))
	fl = fl+list(np.logspace(log10(f3),log10(f4),inf,endpoint=True)) 
	
	SA = []
	for f in fl:
		if f < f1:
			SA.append(a[0]*beta*f)
		elif f < f2:
			SA.append(a[1]*beta)
		elif f < f3:
			SA.append((a[2]*beta-a[3])/f-a[4]*beta+a[5])
		else:
			SA.append(a[6])
			
	rs = ResponseSpectrum(np.array(fl),np.array(SA),abscissa='f',ordinate='sag',xi=xi)
	rs.setLabel('IEEE 693 RRS '+str(xi*100)+'%')
	return rs

def pml(pga,pga_unit='g',xi=0.05,inf=10,gtype='HARD'):

	if pga_unit == 'g':
		ord = 'sag'
		g = 9.81
	else:
		ord = 'sa'
		g = 1.0

	gtype = gtype.upper()
	if gtype not in ('HARD','MEDIUM','SOFT'):
		raise ValueError('Unsupported ground type in call to pml function.')
	
	params = {'HARD' : [0.038, 6.203, 4.89, 1.15, 3.64, 0.78, 3.21, 0.26, 12.0, 40.0],
			  'MEDIUM' : [0.056, 3.465, 5.34, 1.19, 4.0, 0.85, 3.03, 0.39, 10.0, 33.0],
			  'SOFT' : [0.067, 3.335, 5.54, 1.26, 4.28, 0.91, 3.48, 0.46, 8.0, 33.0]}
	
	r0 = params[gtype][0]
	r1 = params[gtype][1]
	a1 = params[gtype][2]
	a2 = params[gtype][3]
	v1 = params[gtype][4]
	v2 = params[gtype][5]
	d1 = params[gtype][6]
	d2 = params[gtype][7]
	pgv = r0*pga*g
	pgd = r1*pgv**2/(pga*g)
	psa = pga*(a1-a2*log(100*xi))
	psv = pgv*(v1-v2*log(100*xi))
	psd = pgd*(d1-d2*log(100*xi))
	
	f1 = 0.1
	f2 = psv/(2*pi*psd)
	f3 = psa*g/(2*pi*psv)
	f4 = params[gtype][8]
	f5 = params[gtype][9]
	
	fl = list(np.linspace(f1,f2,inf,endpoint=False))
	fl = fl+list(np.linspace(f2,f3,inf,endpoint=False))
	fl = fl+list(np.linspace(f3,f4,inf,endpoint=False))
	fl = fl+list(np.linspace(f4,f5,inf,endpoint=False))
	fl = fl+list(np.linspace(f5,100.0,inf,endpoint=True))
	
	m = log(psa/pga)/log(f4/f5)
	b = log(psa)-m*log(f4)
	SA = []		 

	for f in fl:
		if f < f2:
			SA.append(psd*(2*pi*f)**2/g)
		elif f < f3:
			SA.append(psv*2*pi*f/g)
		elif f < f4:
			SA.append(psa)
		elif f < f5:
			SA.append(exp(m*log(f)+b))
		else:
			SA.append(pga)
			
	rs = ResponseSpectrum(np.array(fl),np.array(SA),abscissa='f',ordinate=ord,xi=xi)
	rs.setLabel('PML GT '+str(gtype)+' '+str(xi*100)+'%')
	return rs

def eur_vhr(pga,sa_unit='g',inf=10):
	'''
	Create a response spectrum in accordance with the draft EUR Very Hard Rock spectrum (model 1).
	This function will generate a spectrum in the range T = [0.005;10] sec or f = [0.1;200] Hz.
	
	Parameters
	----------
	pga : float
		Peak gropund acceleration
	sa_unit : string, optional, ``['g'|'m/s**2']``
		Unit of the PGA. Default value 'g'.
	inf : integer, optional
		Number of points in each interval resulting in a total of ``ndat = 4*inf`` points. Default value 10.
	'''
	
	# Define spectral parameters 
	psa = 2.27*pga
	Ta = 0.005
	Tb = 0.03
	Tc = 0.20
	Td = 2.5
	
	Tl = list(np.logspace(log10(Ta),log10(Tb),num=inf,endpoint=False))
	Tl = Tl+list(np.logspace(log10(Tb),log10(Tc),num=inf,endpoint=False))
	Tl = Tl+list(np.logspace(log10(Tc),log10(Td),num=inf,endpoint=False))
	Tl = Tl+list(np.logspace(log10(Td),log10(10),num=inf,endpoint=True))
	
	SA = []
	for T in Tl:
		if T < Tb:
			a = log(pga/psa)/log(Ta/Tb)
			b = pga/Ta**a
			SA.append(b*T**a)
		elif T < Tc:
			SA.append(psa)
		elif T < Td:
			SA.append(psa*Tc/T)
		else:
			SA.append(psa*Tc*Td/T**2)

	if sa_unit == 'g':
		ord = 'sag'
	else:
		ord = 'sa'
	
	rs = ResponseSpectrum(np.array(Tl),np.array(SA),abscissa='T',ordinate=ord,xi=0.05)	
	rs.setLabel('EUR VHR Model 1 (5%)')
	return rs