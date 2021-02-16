"""
Package: Qtools
Module: response_spectrum
(C) 2020-2021 Andreas H. Nielsen
See README.md for further details.
"""

import numpy as np
import matplotlib as mpl
import copy
import functools
import multiprocessing
from itertools import combinations, permutations
from scipy.integrate import odeint
from scipy import interpolate
from math import exp, log, log10, pi, sqrt, floor, sin, cos, isclose
from pathlib import Path
from qtools.systems import vis1, sld
from qtools import config
try:
	from .dlls import auxforsubs
except ModuleNotFoundError:
	config.vprint('WARNING: module auxforsubs not found. '
			   'Qtools will use the slower Python solver for ODE integration.')
	USE_FORTRAN_SUBS = False
except ImportError:
	config.vprint('WARNING: ImportError occured when attempting to import '
			   'module auxforsubs. '
			   'Qtools will use the slower Python solver for ODE integration.')
	USE_FORTRAN_SUBS = False
else:
	USE_FORTRAN_SUBS = True

@config.set_module('qtools')
class ResponseSpectrum:
	"""
	General class for response spectra.

	Parameters
	----------
	x : 1D NumPy array
		A monotonically increasing array of frequency or period values.
	y : 1D NumPy array
		Spectral ordinates (acceleration, velocity or displacement). This must
		have same length as `x`.
	abscissa : {'f', 'T'}, optional
		Specify quantity contained in x, with 'f' = frequency and 'T' = period.
		Default 'f'.
	ordinate : {'sag', 'sa', 'sv', 'sd'}, optional
		Specify quantity contained in `y` with:

		* 'sag' = acceleration (units of g)
		* 'sa' = acceleration (units of |m/s2|)
		* 'sv' = velocity (units of m/s)
		* 'sd' = displacement (units of m)

		Default 'sag'.
	xi : float, optional
		Damping ratio. Default 0.05.
	label : str, optional
		Label to use in the legend when plotting the response spectrum.
		Default '_nolegend_'.
	fmt : str, optional
		The line format to be used when plotting this spectrum (see notes on
		the `fmt` parameter in `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_.
		With the default value, ``fmt = ''`` (an empty string), the spectrum
		will be plotted with the default :mod:`matplotlib.pyplot` line styles.

	Attributes
	----------
	f : 1D NumPy array
		Frequency in units of Hz (1/s).
	fmt : str
		See above under Parameters.
	g : float
		Value of the acceleration due to gravity.
	label : str
		See above under Parameters.
	ndat : integer
		Number of elements in `sa`, `sv`, `sd`, `T`, and `f`.
	sa : 1D NumPy array
		Spectral acceleration in units of g.
	sd : 1D NumPy array
		Spectral displacement in units of m.
	sv : 1D NumPy array
		Spectral velocity in units of m/s.
	T : 1D NumPy array
		Period in units of s (:math:`T = 1/f`).
	xi : float
		Damping ratio.

	Notes
	-----
	The following conventions for units are followed:

	* Each instance of ResponseSpectrum is defined in terms of standard units.
	* The standard spatial unit is metres.
	* The standard time unit is seconds.
	* The standard unit for spectral acceleration is g
	  (with g = 9.81 |m/s2|).
	* Other units are not supported, but may be used at the discretion of the
	  user.

	When `rs` is an instance of ResponseSpectrum, ``str(rs)`` will return a
	string representation of response spectrum `rs`; this will also work with
	the built-in functions ``format(rs)`` and ``print(rs)``.

	Multiplication is supported: `a*rs` multiplies the spectral ordinates of
	`rs` with `a` where `a` must be an integer or a float.

	"""

	def __init__(self, x, y, abscissa='f', ordinate='sag', xi=0.05,
			  label='_nolegend_', fmt=''):

		# Check the supplied arguments
		if type(x) != np.ndarray or type(y) != np.ndarray:
			raise TypeError('The two arguments x and y must be 1D NumPy arrays.')
		if len(x) != len(y):
			raise ValueError('The two arguments x and y, which must '
					'have the same length, have lengths {} and {}'.format(len(x),len(y)))
		if min(np.diff(x)) <= 0:
			raise ValueError('The argument x must be an array of monotonically '
					'increasing values. This error typically arises when two '
					'closely spaced frequencies become identical due to rounding.')

		if abscissa == 'f':
			self.f = x
			self.T = self._invert(x)
		elif abscissa == 'T':
			x = np.flip(x)
			y = np.flip(y)
			self.T = x
			self.f = self._invert(x)
		else:
			raise ValueError('In class ResponseSpectrum, the abscissa must be '
					'frequency (f) or period (T).')

		self.g = 9.81
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
		elif ordinate == 'ei':
			self.sa = y
		else:
			raise ValueError('In class ResponseSpectrum, the ordinate must be sag, sa, sv, or sd.')

		self.xi = xi
		self.ndat = len(y)
		self.k = 0
		self.label = label
		self.fmt = fmt
		self.linestyle = ''
		self.color = ''

	def __call__(self, f, kind = 'loglog'):
		"""
		Evaluate a response spectrum at a particular frequency. Extrapolations
		are allowed.

		Parameters
		----------
		f : float
			Frequency at which to evaluate the response spectrum.
		kind : str, optional
			Manner of interpolation. The default is 'loglog'. See
			:meth:`interp` for further information.

		Returns
		-------
		sa : float
			The spectral acceleration at frequency `f`.

		"""
		if kind == 'linlin':
			sa = np.interp(f,self.f,self.sa)
		elif kind == 'loglin':
			sa = np.interp(log(f),np.log(self.f),self.sa)
		elif kind == 'loglog':
			sa = exp(np.interp(log(f),np.log(self.f),np.log(self.sa)))
		elif kind == 'linlog':
			sa = exp(np.interp(f,self.f,np.log(self.sa)))
		else:
			raise ValueError('You cannot evaluate a response spectrum with '
					' kind = {}.'.format(kind))
		return sa

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
		return self.label

	def __mul__(self,factor):
		if (type(factor) == float or type(factor) == int):
			product = copy.deepcopy(self)
			product.sa *= factor
			if type(self) is ResponseSpectrum:
				product.sv *= factor
				product.sd *= factor
			return product
		else:
			return NotImplemented

	def __rmul__(self,factor):
		return self.__mul__(factor)

	def _invert(self,a):
		if isclose(a[0],0):
			b = np.empty_like(a)
			b[0] = 10**(-floor(log10(a[1])))
			b[1:] = 1/a[1:]
		else:
			b = 1/a
		return b

	def setLabel(self,label):
		"""Sets the label of the spectrum."""
		self.label = str(label)

	def setLineFormat(self, fmt, color=''):
		"""Sets the `fmt` attribute equal to the supplied argument. Optionally,
		define the colour of the line."""

		self.fmt = fmt
		if mpl.colors.is_color_like(color):
			self.color = color

	def savers(self, fname, abscissa='f', ordinate='sag', fmt='%.18e',
			   delimiter=' ', newline='\n', header='_default_', footer='',
			   comments='# '):
		"""
		Save the response spectrum to text file. This function uses
		:func:`numpy.savetxt` for performing the output operation. See
		documentation for :func:`numpy.savetxt` to understand the full
		functionality.

		Parameters
		----------
		fname : str
			Filename or file handle. See :func:`numpy.savetxt`.
		abscissa : {'f', 'T'}, optional
			Output frequencies ('f') or periods ('T'). Default 'f'.
		ordinate : {'sa', 'sag', 'sv', 'sd', 'ei', 'all'}, optional
			Output spectral accelerations ('sa' or 'sag'), velocities ('sv'),
			displacements ('sd'), input energies ('ei') or all quantities
			('all'). Note: 'sa' and 'sag' will both result in accelerations
			in units of g. Default 'sag'.
		fmt : str or sequence of strs, optional
			See :func:`numpy.savetxt`.
		delimiter : str, optional
			String or character separating columns.
		newline : str, optional
			String or character separating lines.
		header : str, optional
			If `header` is not specified, a default header will be generated
			by this function. Set ``header = ''`` to avoid any header being
			written. All other parameter values will be passed directly to
			:func:`numpy.savetxt`.
		footer : str, optional
			String that will be written at the end of the file.
		comments : str, optional
			String that will be prepended to the `header` and `footer` strings,
			to mark them as comments.
		"""

		if header=='_default_':
			head1 = 'Response spectrum {} saved by Qtools v. {}\n'.format(self.label,config.version)
			head2 = 'Damping level = {}'.format(self.xi)
			header = head1+head2

		if abscissa=='f' and ordinate[0:2]=='sa' or ordinate=='ei':
			out = np.array([self.f,self.sa])

		elif abscissa=='T' and ordinate[0:2]=='sa' or ordinate=='ei':
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
			header += '\nPeriod, SA, SV, SD'

		else:
			config.vprint('WARNING: could not recognise the parameters abscissa '
				 '= {} and/or ordinate = {}'.format(abscissa,ordinate))
			config.vprint('No data has been save to file')
			return

		np.savetxt(fname,out.T,fmt=fmt,newline=newline,delimiter=delimiter,
					   header=header,footer=footer,comments=comments)

	def interp(self, fn, kind = 'loglin', merge=True, eps=0.001,
			sd_extrapolate=True, mutate=True):
		"""
		Interpolate a response spectrum at the frequencies given as the first
		argument to the function.

		Parameters
		----------
		fn : 1D NumPy array
			New frequencies (the spectrum will be interpolated at these
			frequencies)
		kind : str, optional
			Type of interpolation. The following options are supported:

			* 'linlin' - linear x- and y-axis.
			* 'loglin' - logarithmic x-axis, linear y-axis (default).
			* 'loglog' - logarithmic x- and y-axis.
			* 'linlog' - linear x-axis, logarithmic y-axis.

		merge : bool, optional
			Set `merge=True` to merge the new frequencies with the existing
			frequencies of the spectrum.
			Set `merge=False` to replace the existing frequencies with the
			new frequencies.
		eps : float, optional
			If frequencies are merged, duplicate frequencies defined as
			frequencies that satisfy ``(f2-f1)/f2 < eps`` are combined into
			one average frequency. The default value of `eps` is 0.001. The
			average value is then rounded to `n` number of digits
			with `n` computed as ``round(log10(1000/(eps*f2)))``.
		sd_extrapolate : bool, optional
			Manner of extrapolation for ``f < self.f[0]`` (see implementation
			notes below). Default `True`.
		mutate : bool
			If True, replace the original data in the response spectrum with
			new interpolated values. If False, keep original values.

		Returns
		-------
		sa : 1D NumPy array
			The interpolated spectral accelerations.

		Notes
		-----
		Spectral accelerations are interpolated in a lin-lin, log-lin, lin-log,
		or log-log co-ordinate system depending on the value of `kind`.
		The method will extrapolate for ``f < self.f[0]`` (where ``self.f``
		refers to the frequency array of the response spectrum being operated
		on) and for ``f > self.f[-1]``.
		It is assumed that the spectral acceleration is	constant for
		``f > self.f[-1]``.
		If `sd_extrapolate` is True, then it is assumed that the spectral
		displacement is constant for ``f < self.f[0]``.
		If `sd_extrapolate` is False, then it is assumed that the
		spectral acceleration is constant for ``f < self.f[0]``.
		"""
		if merge:
			fn = np.sort(np.append(self.f,fn))
			# Remove duplicate frequencies within a precision of eps (on a log scale)
			ft = np.zeros_like(fn)
			ft[0] = fn[0]
			j = 0
			for i in range(1,np.size(fn)):
				if (fn[i]-ft[j])/fn[i] < eps:
					ft[j] = round(sqrt(fn[i]*ft[j]),round(log10(1000/(eps*fn[i]))))
				else:
					ft[j+1] = fn[i]
					j += 1
			fn = ft[0:j+1]

		if kind == 'linlin':
			sa = np.interp(fn,self.f,self.sa)
		elif kind == 'loglin':
			sa = np.interp(np.log(fn),np.log(self.f),self.sa)
		elif kind == 'loglog':
			sa = np.exp(np.interp(np.log(fn),np.log(self.f),np.log(self.sa)))
		elif kind == 'linlog':
			sa = np.exp(np.interp(fn,self.f,np.log(self.sa)))
		else:
			raise ValueError('In class method interp, kind = {} is not supported'.format(kind))

		if sd_extrapolate:
			for i in range(np.size(fn)):
				if fn[i] < self.f[0]:
					sa[i] = self.sd[0]*(2*pi*fn[i])**2/self.g
				else:
					break

		if mutate:
			self.f = fn
			self.T = self._invert(self.f)
			omega = 2*pi*self.f
			self.sa = sa
			if type(self) is ResponseSpectrum:
				self.sv = self.g*self.sa/omega
				self.sd = self.sv/omega
			self.ndat = len(sa)
			self.k = 0

		return sa

@config.set_module('qtools')
def meanrs(rslist, kind='loglin', sd_extrapolate=True, label='_nolegend_', 
		   fmt=''):
	"""
	Compute the mean of two or more response spectra.

	Parameters
	----------
	rslist : list or tuple
		A sequence containing two or more instances of ResponseSpectrum.
	kind : str, optional
		If the response spectrum supplied in `rslist` have different frequency
		values, interpolation will be used to determine spectral accelerations
		based on a common set of frequencies. The `kind` parameter determines
		the interpolation procedure. See the :meth:`.interp` method for more
		information.
	sd_extrapolate : bool, optional
		Manner of extrapolation for :math:`f < f_{min}` where :math:`f_{min}`
		is the minimum frequency in a frequency array (whichever requires
		extrapolation). See the :meth:`.interp` method for more
		information. Default True.
	label : str, optional
		Parameter passed directly to the constructor of 
		:class:`ResponseSpectrum`.
	fmt : str, optional
		Parameter passed directly to the constructor of 
		:class:`ResponseSpectrum`.

	Returns
	-------
	rs : an instance of ResponseSpectrum
		The mean response spectrum.
	"""

	# Are the frequencies of the supplied spectra the same?
	n0 = rslist[0].ndat
	f0 = rslist[0].f
	fAllEqual = np.all([n0 == rs.ndat for rs in rslist[1:]])
	if fAllEqual:
		fAllEqual = np.all([np.allclose(f0,rs.f) for rs in rslist[1:]])

	# If frequencies are not the same, determine a common set of frequencies
	if not fAllEqual:
		f0 = np.concatenate([rs.f for rs in rslist])
		f0 = np.unique(f0)

	# Are the damping ratios the same?
	xi0 = rslist[0].xi
	xiAllEqual = np.all([xi0 == rs.xi for rs in rslist[1:]])
	if not xiAllEqual:
		config.vprint('WARNING: At least two response spectra supplied to '
				'the meanrs function have different damping ratios.')

	allsa = np.empty((len(rslist),len(f0)))
	for i,rs in enumerate(rslist):
		if fAllEqual:
			allsa[i,:] = rs.sa
		else:
			allsa[i,:] = rs.interp(f0, kind=kind, merge=False,
				  sd_extrapolate=sd_extrapolate, mutate=False)

	sa = np.mean(allsa, axis=0)

	return ResponseSpectrum(f0, sa, xi=xi0, label=label, fmt=fmt)


@config.set_module('qtools')
def envelope(rslist, option=0, kind='loglin', radp=False,
			  sd_extrapolate=True):
	"""
	Compute the envelope of two or more response spectra.

	Parameters
	----------
	rslist : list
		A list containing two or more instances of ResponseSpectrum. This
		first argument can also be a single instance of ResponseSpectrum (for
		backwards compatibility only).
	option : int, optional
		Deprecated since version 1.1. Will be removed in version 2.0. This
		parameter is not used for anything.
	radp : bool
		If True, retain all data points. If False, simplify straight lines
		in a `kind` co-ordinate system. See notes below. Default False.
	kind : {'loglin', 'loglog'}, optional
		Type of interpolation used to determine values between data points.
		Only the following options are valid:

		* 'loglin' - logarithmic x-axis, linear y-axis (default).
		* 'loglog' - logarithmic x-axis and y-axis.

	sd_extrapolate : bool, optional
		Manner of extrapolation for :math:`f < f_{min}` where :math:`f_{min}`
		is the minimum frequency in the frequency arrays of `rs1` or `rs2`
		(whichever requires extrapolation). See also notes below.
		Default True.

	Returns
	-------
	rs : an instance of ResponseSpectrum
		The envelope of all response spectra provided as arguments to the
		function.

	Notes
	-----
	The default behaviour of this function is to delete intermediate data
	points within segments of straight lines in a `kind` co-ordinate system.
	For that reason, enveloping multiple spectra with ``kind = 'loglin'``
	followed by plotting on a log-log graph can lead to unexpected results.
	For the same reason, enveloping with ``kind = 'loglog'`` followed by
	plotting on a lin-log graph can also lead to unexpected results.
	Set ``radp = True`` to avoid straight line simplification and retain all
	data points.

	Spectral accelerations are interpolated and/or extrapolated as necessary
	to compute the envelope over the entire frequency range.
	The method :meth:`.interp` is used for this purpose.
	The parameters `kind` and `sd_extrapolate` are passed directly to
	:meth:`.interp`. That is, when extrapolation is performed,

	* it is assumed that the spectral acceleration is constant for
	  :math:`f > f_{max}` where :math:`f_{max}` is the maximum frequency in the
	  frequency arrays of `rs1` or `rs2` (whichever requires extrapolation);
	* when ``sd_extrapolate is True``, it is assumed that the spectral
	  displacement is constant for :math:`f < f_{min}`;
	* when ``sd_extrapolate is False``, it is assumed that the spectral
	  acceleration is constant for :math:`f < f_{min}`.
	"""

	if kind not in ('loglin','loglog'):
		raise ValueError('The parameter kind must be either \'loglin\' or \'loglog\'')

	# Define a function that returns the envelope of two spectra
	def env2rs(rsa,rsb):
		# Find intersections
		k = 0
		fi = []
		for i in range(len(rsa.f)-1):
			for j in range(k,len(rsb.f)-1):
				if rsb.f[j] >= rsa.f[i+1]: break
				if rsb.f[j+1] <= rsa.f[i]: k += 1; continue
				coords = _intersection(rsa.f[i:i+2], rsa.sa[i:i+2], rsb.f[j:j+2],
						   rsb.sa[j:j+2], kind=kind)
				if coords[0] > 0:
					fi.append(coords[0])
		# Creat an ordered list of all frequencies, removing duplicates
		f0 = np.unique(np.concatenate((rsa.f,rsb.f,fi)))
		# Interpolate at the new frequencies
		saa = rsa.interp(f0, kind=kind, merge=False,
				   sd_extrapolate=sd_extrapolate, mutate=False)
		sab = rsb.interp(f0, kind=kind, merge=False,
				   sd_extrapolate=sd_extrapolate, mutate=False)
		# Envelope
		sa0 = np.fmax(saa,sab)
		if radp:
			f1 = f0
			sa1 = sa0
		else:
			# Remove redundant points (i.e. points on a straight line)
			f1 = []
			sa1 = []
			if kind == 'loglin':
				# Adding 1 just ensures that a1 != a0 in the first loop.
				a0 = (sa0[1]-sa0[0])/(log(f0[1]/f0[0]))+1
			else:
				# Adding 1 just ensures that a1 != a0 in the first loop.
				a0 = log(sa0[1]/sa0[0])/(log(f0[1]/f0[0]))+1
			for i in range(len(f0)-1):
				if kind == 'loglin':
					a1 = (sa0[i+1]-sa0[i])/(log(f0[i+1]/f0[i]))
				else:
					a1 = log(sa0[i+1]/sa0[i])/(log(f0[i+1]/f0[i]))
				if not isclose(a1,a0,abs_tol=1E-9):
					# Add data point
					a0 = a1
					f1.append(f0[i])
					sa1.append(sa0[i])
			f1.append(f0[-1])
			sa1.append(sa0[-1])
		# Finally, check damping
		if rsa.xi != rsb.xi:
			config.vprint('WARNING: an envelope is created of two spectra with'
				 ' different damping ratios.')
			config.vprint('The damping ratio of the new enveloping spectrum is'
					' defined as {}.'.format(rs1.xi))
		return ResponseSpectrum(np.array(f1),np.array(sa1),xi=rsa.xi)

	rs0 = rslist[0]
	# Loop over and envelope all response spectra
	for rs1 in rslist[1:]:
		rs0 = env2rs(rs0,rs1)

	return rs0

@config.set_module('qtools')
def peakbroad(rs, df=0.15, df_sign='plus/minus', peak_cap=False, truncate=True):
	"""Broaden a spectrum.

	Parameters
	----------
	rs : an instance of class ResponseSpectrum
		Input spectrum for broadening operation.
	df : float, optional
		The relative bandwidth for broadening. The default value
		(``df = 0.15``) corresponds to 15% peak broadening. The value assigned
		to `df` must be positive. A value ``df = 0`` results in no broadening.
	df_sign :  {'plus', 'minus', 'plus/minus'}, optional,
		The sign implied on parameter `df`. If for example
		``df_sign = 'minus'``, the function will undertake peak broadening in
		the negative direction only.
	peak_cap : bool, optional.
		Cap the peaks before the broadening operation in accordance with
		ASCE 4-16, 6.2.3(b). Default `False`.
	truncate : bool, optional
		Limit the minimum and maximum frequencies of the broadened spectrum to
		those of the input spectrum. Default `True`.

	Returns
	-------
	rs : an instance of class ResponseSpectrum
		The broadened copy of the input spectrum.

	Notes
	-----
	Input energy is not broadened. The new spectrum will have zero input
	energy. The peak broadening function assumes that the spectrum is a series
	of straight lines on a semi-log graph (with a logarithmic frequency scale).
	If the spectrum is better represented by a series of straight lines on a
	log-log graph, use the class method :meth:`.ResponseSpectrum.interp` with
	``kind = 'loglog'`` to generate as many new points as necessary before
	broadening.
	"""

	# Split the spectrum into segments
	sgn0 = np.sign(rs.sa[1]-rs.sa[0])
	fseg = []; saseg = []
	segs = []; sgns = []
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
				segs.append([np.array(fseg),np.array(saseg)])
				sgns.append(sgn0)
				fseg = [rs.f[i]]
				saseg = [rs.sa[i]]
				sgn0 = sgn1
		else:
			# Finalise last segment
			segs.append([np.array(fseg),np.array(saseg)])
			sgns.append(sgn0)

	# Check that we have at least 2 segments
	if len(sgns) < 2:
		raise ValueError('Cannot peak-broaden a spectrum with {} segments.'.format(len(sgns)))

	# Cap the peaks (if required)
	if peak_cap:
		i = 0
		while i <= len(sgns)-2:
			if sgns[i] == 1 and sgns[i+1] == -1:
				seg_sub, sgn_sub = _peak_cap(segs[i],segs[i+1])
				segs = segs[:i]+seg_sub+segs[i+2:]
				sgns = sgns[:i]+sgn_sub+sgns[i+2:]
			i += 1

	if df > 0:
		# Broaden the segments
		if df_sign == 'plus/minus':
			dfm = df; dfp = df
		elif df_sign == 'plus':
			dfm = 0; dfp = df
		elif df_sign == 'minus':
			dfm = df; dfp = 0
		else:
			raise ValueError('Invalid value assigned to parameter df_sign: {}'.format(df_sign))
		# NB - the following loops modify level 2 list entries in place
		for seg, sgn in zip(segs,sgns):
			if sgn == 1:
				seg[0] = np.append((1-dfm)*seg[0],seg[0][-1])
				seg[1] = np.append(seg[1],seg[1][-1])
			elif sgn == -1:
				seg[0] = np.append(seg[0][0],(1+dfp)*seg[0])
				seg[1] = np.append(seg[1][0],seg[1])

		# Find and insert intersections
		for sa,sb in combinations(segs,2):
			xa = sa[0]; ya = sa[1]
			xb = sb[0]; yb = sb[1]
			if xa[-1] <= xb[0] or xa[0] >= xb[-1] or max(ya) <= min(yb) or min(ya) >= max(yb):
				# Segments cannot intersect
				continue
			else:
				for i in range(len(xa)-1):
					for j in range(len(xb)-1):
						coords = _intersection(xa[i:i+2],ya[i:i+2],xb[j:j+2],yb[j:j+2])
						if coords[0] > 0:
							sa[0] = np.insert(xa,i+1,coords[0])
							sa[1] = np.insert(ya,i+1,coords[1])
							sb[0] = np.insert(xb,j+1,coords[0])
							sb[1] = np.insert(yb,j+1,coords[1])
							break
					else:
						# This else clause is invoked if the inner loop was not broken
						continue # I.e. continue with the outer loop
					break # This statement is executed only if inner loop was broken

		# Define a function that determines whether a point lies above or below
		# a straight line in semilog coordinates
		def point_re_line(x,y,xl,yl):
			m = (yl[1]-yl[0])/log(xl[1]/xl[0])
			b = yl[0]-m*log(xl[0])
			fx = m*log(x)+b
			if isclose(y,fx):
				return 0
			elif y < fx:
				return -1
			else:
				return 1

		# Remove points that are enveloped by lines
		# Note: we could have used combinations(segs,2) in the following loop
		# However, this would have doubled up on the inner two loops
		# Since len(list(permutations(segs,2))) = 2*len(list(combinations(segs,2)))
		# no economy is gained by using combinations(segs,2)
		for sa,sb in permutations(segs,2):
			xa = sa[0]; xb = sb[0]
			if len(xa) == 0 or len(xb) == 0:
				# At least one of the two segments has been emptied of points already
				continue
			if xa[-1] <= xb[0] or xa[0] >= xb[-1]:
				# Segment b cannot envelope any points in segment a
				continue
			else:
				inxa = []
				ya = sa[1]; yb = sb[1]
				for i in range(len(xa)):
					for j in range(len(xb)-1):
						if xa[i] >= xb[j] and xa[i] <= xb[j+1]:
							if point_re_line(xa[i],ya[i],xb[j:j+2],yb[j:j+2]) == -1:
								inxa.append(i)
				if len(inxa) > 0:
					sa[0] = np.delete(xa,inxa)
					sa[1] = np.delete(ya,inxa)

	# Re-assemble the spectrum
	fq, ui = np.unique(np.concatenate(tuple(s[0] for s in segs)), return_index=True)
	sa = np.concatenate(tuple(s[1] for s in segs))[ui]

	# Truncate the spectrum such that rs.f[0] <= f <= rs.f[-1]
	if truncate:
		m = 0; n = len(fq)
		for i in range(len(fq)-1):
			if fq[i] <= rs.f[0] and fq[i+1] > rs.f[0]:
				sa[i] = np.interp(log(rs.f[0]),np.log(fq[i:i+2]),sa[i:i+2])
				fq[i] = rs.f[0]
				m = i
			elif fq[i] < rs.f[-1] and fq[i+1] >= rs.f[-1]:
				sa[i+1] = np.interp(log(rs.f[-1]),np.log(fq[i:i+2]),sa[i:i+2])
				fq[i+1] = rs.f[-1]
				n = i+2

		fq = fq[m:n]
		sa = sa[m:n]

	return ResponseSpectrum(fq,sa,xi=rs.xi)

def _intersection(xa,ya,xb,yb,kind='loglin'):
	"""Auxiliary function used by other functions. Determines whether two
	straight lines, a and b, intersect, and returns the coordinates of the
	intersection if they do.

	Parameters
	----------
	xa : array_like
		The x coordinates of line a. This should have exactly two entries.
	ya : array_like
		The y coordinates of line a. This should have exactly two entries.
	xb : array_like
		The x coordinates of line b. This should have exactly two entries.
	yb : array_like
		The x coordinates of line b. This should have exactly two entries.
	kind : {'loglin','loglog'}, optional
		For `kind = 'loglin'` the lines are assumed to be straight lines
		on a semi-logarithmic coordinate system. For `kind = 'loglog'` the
		lines are assumed to be straight lines in a double-logarithmic
		coordinate system. Default 'loglin'.

	Returns
	-------
	coords : tuple
		The (x,y) coordinates of the intersection. If the two lines do not
		intersect, the function returns (0,0).
	"""
	coords = (0,0)
	x1 = log(xa[0]); x2 = log(xa[1])
	x3 = log(xb[0]); x4 = log(xb[1])
	if kind == 'loglin':
		y1 = ya[0]; y2 = ya[1]
		y3 = yb[0]; y4 = yb[1]
	else:
		y1 = log(ya[0]); y2 = log(ya[1])
		y3 = log(yb[0]); y4 = log(yb[1])
	denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
	# Lines are parallel if denom == 0
	if not isclose(denom,0):
		t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom
		u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denom
		if (t >= 0 and t <= 1) and (u >= 0 and u <= 1):
			xi = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
			yi = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
			if kind == 'loglin':
				coords = (exp(xi),yi)
			else:
				coords = (exp(xi),exp(yi))
	return coords

def _peak_cap(left,right):
	"""Auxiliary function used by function peakbroad()
	Cap a peak in accordance with ASCE 4-16, 6.2.3(b). Takes as input two
	arrays, left and right, with:

		* left[0] = an increasing frequency array (left[0][i] < left[0][i+1])
		* left[1] = an ascending acceleration array (left[1][i] < left[1][i+1])
		* right[0] = an increasing frequency array (right[0][i] < right[0][i+1])
		* right[1] = a descending acceleration array (right[0][i] > right[0][i+1])

	It is assumed that left[0][-1] = right[0][0] and left[1][-1] = right[1][0]
	-  i.e. the two segments share the peak point.
	"""

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
		   [np.append(fR85,fR0[saR0 < sa85]),np.append(sa85,saR0[saR0 < sa85])]]
		sgns = [1,0,-1]
	else:
		rsls = [left,right]
		sgns = [1,-1]
	return rsls,sgns

@config.set_module('qtools')
def calcrs(th, ffile=None, nf=200, fmin=0.1, fmax=100., xi=0.05,
		   solver='solode', accuracy='medium', MP=False):
	"""
	Calculate a response spectrum from a time history.

	Parameters
	----------
	th : an instance of class TimeHistory
		An acceleration time history defining the base motion.
	ffile : str, optional
		Read frequency points from file `ffile`. If not specified, the function
		will generate `nf` frequency points between `fmin` and `fmax` equally
		spaced on a logarithmic scale.
	nf : int, optional
		Number of frequency points. Default 200. Ignored if `ffile` is
		specified.
	fmin : float, optional
		Minimum frequency considered. Default 0.1 Hz. Ignored if `ffile` is
		specified.
	fmax : float, optional
		Maximum frequency considered. Default 100.0 Hz. Ignored if `ffile` is
		specified.
	xi : float, optional
		Damping ratio. Default 0.05.
	solver : {'solode', 'odeint'}, optional
		Solution algorithm to be use for the computation. Default 'solode'.
	accuracy : {'low', 'medium', 'high'}, optional
		The required accuracy of the response spectrum. Valid with solver
		'odeint' only. The specified accuracy can affect high frequency values.
	MP : bool, optional
		Use multi-processing to compute the response spectrum. Multi-processing
		is only implemented for the solode solver. The input energy is not
		calculated under multi-processing. Default False.

	Returns
	-------
	rs : an instance of ResponseSpectrum

	Notes
	-----
	The default solver is solode, which is particularly suited to a second
	order linear ordinary differential equation (ODE) with numerical
	data as a forcing function (as opposed to an analytical function).
	The odeint solver is from SciPy's collection of ODE solvers. This solver is
	more general in application. For more information see
	`scipy.integrate.odeint <https://docs.scipy.org/doc/scipy/reference
	/generated/scipy.integrate.odeint.html#scipy.integrate.odeint>`_.
	The odeint solver can be used for verification purposes.
	"""

	# Code involving dml is legacy code which can be ignored.
	dml = 'viscous'

	# Preliminary checks
	if th.ordinate != 'a':
		raise TypeError('The time history used as input to function calcrs must be an acceleration time history.')
	if xi >= 1.0 and solver=='solode':
		raise ValueError('The damping ratio must be less than 1 when using the solode solver.')
	if solver == 'odeint':
		config.vprint('Using solver odeint to compute response spectrum with accuracy = {}.'.format(accuracy))
		if dml != 'viscous':
			config.vprint('NOTE: The damping model is: {}'.format(dml))
	elif solver[-6:] == 'solode':
		config.vprint('Using solver solode to compute response spectrum.')
		if USE_FORTRAN_SUBS:
			solver = 'fsolode'
		else:
			solver = 'psolode'
		if th.dt_fixed:
			config.vprint('The time step is taken as fixed.')
		else:
			config.vprint('The time step is taken as variable.')
	else:
		raise ValueError('{} is not a valid solver'.format(solver))

	if ffile==None:
		# Generate frequency points
		f = np.logspace(log10(fmin),log10(fmax),nf)
	else:
		# Read frequency points from file
		f = np.loadtxt(ffile)
		nf = np.size(f)
		fmax = f[-1]
	if f[0] >= th.fNyq:
		raise ValueError('The lowest frequency of the spectrum is greater than, '
				   'or equal to, the Nyquist frequency of the time history.')

	w = 2*pi*f[f<=th.fNyq]
	y0 = [0.0,0.0]

	if MP:
		processes = max(multiprocessing.cpu_count()-1, 1)
		if processes > 1:
			# Multiple processes
			config.vprint('Will use {} processes to compute spectrum'.format(processes))
			config.vprint('Remember to protect the main module with an if __name__ == \'__main__\': statement')
			config.vprint('WARNING: The input energy is not calculated when using multiple processes.')
			with multiprocessing.Pool(processes=processes) as pool:
				sd = pool.map(functools.partial(_solode, y0, th, xi), w)
		else:
			config.vprint('WARNING: Only one process available; reverting to single processing.')
			MP = False

	if not MP:
		# Single process
		sd = np.empty_like(w)
		for i in range(len(w)):
			if solver == 'odeint':
				if dml == 'viscous':
					fun1 = vis1
				elif dml == 'solid':
					fun1 = sld
				if accuracy == 'low':
					# Low accuracy - for quick and dirty solutions
					sol = odeint(fun1,y0,th.time,args=(w[i],xi,th),atol=0.0001,rtol=0.001)
				elif accuracy == 'medium':
					# The atol and rtol values in the following call seem to
					# provide a reasonable compromise between speed and accuracy
					sol = odeint(fun1,y0,th.time,args=(w[i],xi,th),atol=0.000001,rtol=0.001)
				elif accuracy == 'high':
					# For solutions that display spurious high frequency behaviour, the following call should be tried:
					sol = odeint(fun1,y0,th.time,args=(w[i],xi,th))
				sd[i] = np.amax(np.fabs(sol[:,0]))
			elif solver == 'fsolode':
				sd[i], sol = auxforsubs.solode(y0,th.time,th.data,th.dt_fixed,xi,w[i])
			elif solver == 'psolode':
				sd[i] = _solode(y0,th,xi,w[i])

	# Set spectral accelerations equal to the PGA for f greater than the Nyquist frequency
	if fmax > th.fNyq:
		config.vprint('NOTE: Spectral accelerations above the Nyquist '
				'frequency ({:6.2f} Hz) are assumed equal to the ZPA.'.format(th.fNyq))
		sd = np.append(sd,th.pga/(2*pi*f[sd.size:])**2)

	# Complete the creation of a response spectrum
	config.vprint('------------------------------------------')
	rs = ResponseSpectrum(f, sd, abscissa='f', ordinate='sd', xi=xi)
	rs.setLabel(th.label)
	return rs

@config.set_module('qtools')
def calcrs_cmp(ths, ffile=None, nf=200, fmin=0.1, fmax=100, xi=0.05,
		   cmp='GM', theta=0):
	"""
	Calculate a response spectrum from a time history set and return a
	specified horizontal component.

	Parameters
	----------
	ths : an instance of class TimeHistorySet
		An acceleration time history set defining the base motion. Only
		horizontal motions will be processed. The vertical motion, if included
		in the set, will be ignored.
	ffile : str, optional
		Read frequency points from file `ffile`. If `ffile` is not specified,
		the function will generate `nf` frequency points between `fmin` and
		`fmax` equally spaced on a logarithmic scale.
	nf : int, optional
		Number of frequency points. Default 200. Ignored if `ffile` is
		specified.
	fmin : float, optional
		Minimum frequency considered. Default 0.1 Hz. Ignored if `ffile` is
		specified.
	fmax : float, optional
		Maximum frequency considered. Default 100.0 Hz. Ignored if `ffile` is
		specified.
	xi : float, optional
		Damping ratio. Default 0.05.
	cmp : {'GM', 'MC', 'MD', 'GMRotD50'}, optional
		Desired output component:

		* 'GM' is geometric mean computed for rotation angle `theta`.
		* 'MC' is maximum component, i.e. the maximum spectral acceleration
		  of the two components computed at rotation angle `theta`.
		* 'MD' is maximum direction, i.e. the maximum spectral acceleration
		  for all possible rotations (`theta` is ignored).
		* 'GMRotD50' is the median value of the geometric mean for all
		  possible rotations (`theta` is ignored).

		Default value 'GM'.
	theta : float, optional
		Rotation angle in degrees. The two horiontal base motions in the time
		history set will be rotated by `theta` degrees.

	Returns
	-------
	rs : an instance of ResponseSpectrum
	"""

	# Preliminary checks
	if ths[0].ordinate != 'a':
		raise TypeError('The time history used as input to function calcrs_cmp '
				  'must be an acceleration time history.')
	if xi >= 1.0:
		raise ValueError('The damping ratio must be less than 1 when using the solode solver.')
	if ths[0].dt_fixed:
		config.vprint('The time step is taken as fixed.')
	else:
		config.vprint('The time step is taken as variable.')

	if ffile == None:
		# Generate frequency points
		f = np.logspace(log10(fmin), log10(fmax), nf)
	else:
		# Read frequency points from file
		f = np.loadtxt(ffile)
		nf = np.size(f)
		fmax = f[-1]
	if f[0] >= ths[0].fNyq:
		raise ValueError('The lowest frequency of the spectrum is greater than, '
				   'or equal to, the Nyquist frequency of the time history.')

	w = 2*pi*f
	y0 = [0.0,0.0]

	if cmp == 'GM' or cmp == 'MC' or cmp == 'MD':
		angles = [np.radians(theta)]
	elif cmp == 'GMRotD50':
		angles = np.radians(np.arange(0, 90, step=1))
	else:
		raise ValueError('The option cmp = {} is not supported.'.format(cmp))

	config.vprint('Using solver solode to compute response spectrum for component {}.'.format(cmp))
	ax = ths[0].data; ay = ths[1].data
	sd = np.empty((len(w),len(angles)))
	for i in range(len(w)):
		for j, q in enumerate(angles):
			a0 = ax*cos(q) + ay*sin(q)
			a1 = -ax*sin(q) + ay*cos(q)
			if USE_FORTRAN_SUBS:
				sd0, sol0 = auxforsubs.solode(y0,ths[0].time,a0,ths[0].dt_fixed,xi,w[i])
				sd1, sol1 = auxforsubs.solode(y0,ths[1].time,a1,ths[1].dt_fixed,xi,w[i])
			else:
				ths[0].data = a0; ths[1].data = a1
				sol0 = _solode(y0,ths[0],xi,w[i],peak_resp_only=False)
				sd0 = np.amax(np.fabs(sol0[:,0]))
				sol1 = _solode(y0,ths[1],xi,w[i],peak_resp_only=False)
				sd1 = np.amax(np.fabs(sol0[:,0]))
			if cmp == 'GM':
				sd[i,j] = sqrt(sd0*sd1)
			elif cmp == 'MC':
				sd[i,j] = max(sd0, sd1)
			elif cmp == 'MD':
				sd[i,j] = np.amax(np.sqrt(sol0[:,0]**2 + sol1[:,0]**2))
			elif cmp == 'GMRotD50':
				sd[i,j] = sqrt(sd0*sd1)
		if cmp == 'GMRotD50':
			sd[i,0] = np.median(sd[i,:])

	# Retore original data in the time histories
	ths[0].data = ax; ths[1].data = ay
	# Complete the creation of a response spectrum
	config.vprint('------------------------------------------')
	rs = ResponseSpectrum(f, sd[:,0], abscissa='f', ordinate='sd', xi=xi)
	rs.setLabel(' '.join((ths.label,cmp)))
	return rs

def _solode(y0,th,xi,w,peak_resp_only=True):
	"""
	Solver for a second order linear ordinary differential equation.
	This solver is intended for a single degree-of-freedom system subject to
	base excitation.

	Parameters
	----------
	y0 : tuple or list of floats
		Initial conditions: y0[0] = initial displacement, y0[1] = initial
		velocity
	th : instance of class TimeHistory
		Time history defining the base acceleration
	xi : float
		Damping ratio of the system
	w : float
		Undamped natural frequency of the system (rad/s)
	peak_resp_only : bool, optional
		If True, return only the maximum absolute displacement, otherwise
		return a full response time history. Default True.

	Returns
	-------
	If ``peak_resp_only == True``:
	y : float
		The maximum absolute displacement (spectral displacement)

	If ``peak_resp_only == False``:
	y : NumPy array, shape (len(th.ndat),2)
		Array containing the values of displacement (first column) and
		velocities (second column) for each time point in th.time, with the
		initial values y0 in the first row.
	"""

	wd = w*sqrt(1-xi**2)
	n = th.ndat
	y = np.empty((n,2))
	y[0,:] = y0
	f = -th.data

	def ab(dt,w,wd,xi):
		a = np.empty((2,2))
		b = np.empty((2,2))
		c = cos(wd*dt)
		s = sin(wd*dt)
		e = exp(-xi*w*dt)
		a[0,0] = e*(xi*w*s/wd+c)
		a[0,1] = e*s/wd
		a[1,0] = -e*w*s/sqrt(1-xi**2)
		a[1,1] = e*(c-xi*w*s/wd)
		b[0,0] = -e*((xi/(w*wd)+(2*xi**2-1)/(w**2*wd*dt))*s+(1/w**2+2*xi/(w**3*dt))*c)+2*xi/(w**3*dt)
		b[0,1] = e*((2*xi**2-1)/(w**2*wd*dt)*s+2*xi/(w**3*dt)*c)+1/w**2-2*xi/(w**3*dt)
		b[1,0] = -e*((wd*c-xi*w*s)*((2*xi**2-1)/(w**2*wd*dt)+xi/(wd*w))
			   -(wd*s+xi*w*c)*(1/w**2+2*xi/(w**3*dt)))-1/(w**2*dt)
		b[1,1] = -e*(-(wd*c-xi*w*s)*(2*xi**2-1)/(wd*w**2*dt)+(wd*s+xi*w*c)*2*xi/(w**3*dt))+1/(w**2*dt)
		return (a,b)

	if th.dt_fixed:
		dt = th.dt
		a,b = ab(dt,w,wd,xi)
		for i in range(n-1):
			y[i+1,:] = a@y[i,:]+b@[f[i],f[i+1]]
	else:
		t = th.time
		dt0 = 0
		for i in range(n-1):
			dt = t[i+1]-t[i]
			# Ignore very small variations in time step
			if not isclose(dt,dt0):
				a,b = ab(dt,w,wd,xi)
			dt0 = dt
			y[i+1,:] = a@y[i,:]+b@[f[i],f[i+1]]

	if peak_resp_only:
		return np.amax(np.fabs(y[:,0]))
	else:
		return y

@config.set_module('qtools')
def loadrs(sfile, abscissa='f', ordinate='sag', length='m', xi=0.05,
		   delimiter=None, comments='#', skiprows=0, label='_nolegend_', 
		   fmt=''):
	"""
	Load a response spectrum from a text file. The x-values (abscissae) must
	be in the first column,	and the y-values (ordinates) must in the second
	column of the file. Note that the specification of `xi` in this function
	has no effect on the ordinates. The input spectrum may be in units of L,
	L/s, L/s\ :sup:`2` (where L is the length dimension, e.g. metres) or in
	units of g. The following are examples of valid specifications:

		* ``ordinate = 'sa'`` and ``length = 'm'``: the ordinates are defined
		  in units of |m/s2|.
		* ``ordinate = 'sag'`` and ``length = 'm'``: the ordinates are defined
		  in units of g.
		* ``ordinate = 'sv'`` and ``length = 'cm'``: the ordinates are defined
		  in units of cm/s.
		* ``ordinate = 'sd'`` and ``length = 'mm'``: the ordinates are defined
		  in units of mm/s.

	Parameters
	----------
	sfile : str
		The path and name of the input file.
	abscissa : {'f', 'T'}, optional
		The physical quantity of the data in the first column of the input
		file with 'f' = frequency and 'T' = period. Default 'f'.
	ordinate : {'sa', 'sag', 'sv', 'sd'}, optional
		The physical quantity of the data in the second column of the input
		file. Default 'sa'. Note that:

		* 'sa' = spectral acceleration (unit L/s\ :sup:`2`)
		* 'sag' = spectral acceleration (unit g)
		* 'sv' = spectral velocity (unit L/s)
		* 'sd' = spectral displacement (unit L)

	length : {'m', 'dm', 'cm', 'mm'}, optional
		Unit of length dimension L.

	Returns
	-------
	rs : an instance of class ResponseSpectrum

	Notes
	-----
	See :func:`numpy.loadtxt` for further information on parameters `delimiter`,
	`comments` and `skiprows`. See :class:`ResponseSpectrum` for information
	on parameters `label` and `fmt`.
	"""

	lf = {'m': 1, 'dm': 10, 'cm': 100, 'mm': 1000}
	inp = np.loadtxt(sfile, delimiter=delimiter, comments=comments, skiprows=skiprows)
	x = inp[:,0]
	y = inp[:,1]/lf[length]
	rs = ResponseSpectrum(x ,y , abscissa=abscissa, ordinate=ordinate, xi=xi, 
					   fmt=fmt)
	if label=='_nolegend_':
		# Set label based on filename (removing the file path and the extension, if any)
		rs.setLabel(Path(sfile).stem)
	else:
		rs.setLabel(label)
		
	return rs

@config.set_module('qtools')
def dmpinterp(rs1, rs2 , xi, method=2, kind='loglin', sd_extrapolate=True):
	"""Interpolates two response spectra at a new damping level.

	Parameters
	----------
	rs1 : an instance of class ResponseSpectrum
		The first response spectrum to be considered (with damping level
		`rs1.xi`).
	rs2 : an instance of class ResponseSpectrum
		The second response spectrum to be considered (with damping level
		`rs2.xi`, which must be different from `rs1.xi`).
	xi : float
		Damping level for the new response spectrum.
	method : {1, 2}, optional
		Method of interpolation between two damping values. See notes below.
	kind : {'loglin', 'loglog'}, optional
		Type of interpolation used to determine spectral acceleration values
		between two frequencies for a given level of damping.
		Only the following options are valid:

		* 'loglin' - logarithmic x-axis, linear y-axis (default).
		* 'loglog' - logarithmic x-axis and y-axis.

	sd_extrapolate : bool, optional
		Manner of extrapolation for :math:`f < f_{min}` where :math:`f_{min}`
		is the minimum frequency in the frequency arrays of `rs1` or `rs2`
		(whichever requires extrapolation). See also :func:`.envelope`.

	Returns
	-------
	rs : An instance of class ResponseSpectrum
		A new response spectrum interpolated at damping `xi`.

	Notes
	-----
	This function will first interpolate `rs1` and `rs2` individually at a
	common set of frequencies based on the parameters `kind` and
	`sd_extrapolate`. Then the function will interpolate between `rs1` and `rs2`
	and thereby compute a new response spectrum at the intermediate damping
	level `xi`.

	Two methods are recommended for interpolating response spectra at
	intermediate damping values. The first is linear interpolation on the
	natural logarithms of the damping values (set ``method = 1`` to use this
	method). The second is considered more accurate (`ASCE 4-16`_, C6.2.4)
	and is based on a random vibration approach (set ``method = 2`` to use this
	method). The details of this approach are described by `Preumont (1988)`_.
	Thus, method 1 implements `ASCE 4-98`_, (Eq. 2.2-1), and method 2
	implements `ASCE 4-16`_, (6.2).

	This function can also be used for extrapolation although this is not
	recommended.

	References
	----------
	.. _ASCE 4-98:

	ASCE 4-98, *Seismic Analysis of Safety-Related Nuclear
	Structures and Commentary*, American Society of Civil Engineers, 1999.

	.. _ASCE 4-16:

	ASCE/SEI 4-16, *Seismic Analysis of Safety-Related Nuclear
	Structures and Commentary*, American Society of Civil Engineers, 2017.

	.. _Preumont (1988):

	Preumont, A., 1988: "Application of the random vibration approach in the
	seismic analysis of LMFBR structures", *Nuclear Science and Technology*,
	Commission of the European Communities, Luxembourg.
	"""
	# Check damping levels
	if isclose(rs1.xi,rs2.xi):
		raise ValueError('The damping levels of rs1 and rs2 must be different.')
	if xi < min(rs1.xi,rs2.xi) or xi > max(rs1.xi,rs2.xi):
		config.vprint('WARNING: function dampinterp is used for extrapolation.')
	if max(rs1.xi,rs2.xi) > 3*min(rs1.xi,rs2.xi):
		config.vprint('WARNING: the maxmum damping ratio is more than 3 times the '
				'minimum damping ratio, which outside ASCE 4 recommended bounds.')
	# Create copies of rs1 and rs1 to avoid changing the values of their attributes
	rsa = copy.deepcopy(rs1)
	rsb = copy.deepcopy(rs2)
	# Find intersections (hopefully none!)
	k = 0
	fi = []
	for i in range(len(rsa.f)-1):
		for j in range(k,len(rsb.f)-1):
			if rsb.f[j] >= rsa.f[i+1]: break
			if rsb.f[j+1] <= rsa.f[i]: k += 1; continue
			coords = _intersection(rsa.f[i:i+2],rsa.sa[i:i+2],rsb.f[j:j+2],rsb.sa[j:j+2],kind=kind)
			if coords[0] > 0:
				config.vprint('WARNING: The two response spectra supplied as '
				  'arguments to function dmpinterp intersect at f = {}.'.format(coords[0]))
				fi.append(coords[0])
	# Interpolate spectra at a common set of frequencies
	f0 = np.unique(np.concatenate((rsa.f,rsb.f,fi)))
	rsa.interp(f0, kind=kind, merge=False, sd_extrapolate=sd_extrapolate)
	rsb.interp(f0, kind=kind, merge=False, sd_extrapolate=sd_extrapolate)
	# Interpolate at damping levels
	if method == 1:
		sa = rsa.sa+(rsb.sa-rsa.sa)*log(xi/rsa.xi)/log(rsb.xi/rsa.xi)
	else:
		sa = np.sqrt(rsb.sa**2+(rsa.sa**2-rsb.sa**2)*rsa.xi/xi*(xi-rsb.xi)/(rsa.xi-rsb.xi))

	return ResponseSpectrum(f0,sa,xi=xi)

@config.set_module('qtools')
def ec8_2004(pga, pga_unit='g', xi=0.05, inf=10, stype=1, gtype='A', q=1.0,
			 option='HE'):
	"""
	Create a response spectrum in accordance with BS EN 1998-1:2004+A1:2013.
	This function will generate a spectrum in the range `T` = [0.01;10] sec or
	`f` = [0.1;100] Hz.

	Parameters
	----------
	pga : float
		Peak ground acceleration.
	pga_unit : {'g', 'm/s**2'}, optional
		Unit of the pga value. Default value 'g'.
	xi : float, optional
		Damping ratio. Default value 0.05.
	inf : int, optional
		Number of points in each interval resulting in a total of `4*inf`
		points. Default value 10.
	stype : {1, 2}, optional
		Specify Type 1 or 2 spectrum. Default value 1.
	gtype : {'A', 'B', 'C', 'D', 'E'}, optional
		Ground type. Default value 'A'.
	q : float, optional
		Behaviour factor.
	option : {'HE', 'HD', 'VE', 'VD'}, optional
		Type of spectrum: Horizontal / Vertical, Elastic / Design.
		The default value, 'HE', is the horizontal elastic spectrum.

	Returns
	-------
	rs : an instance of class ResponseSpectrum
	"""

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
		config.vprint('Warning: option HD has not been implemented yet.')
	elif option=='VE':
		config.vprint('Warning: option VE has not been implemented yet.')
	elif option=='VD':
		config.vprint('Warning: option VD has not been implemented yet.')
	else:
		raise ValueError('Unsupported option in call to ec8_2004: option = {}'.format(option))

	if pga_unit == 'g':
		ord = 'sag'
	else:
		ord = 'sa'

	rs = ResponseSpectrum(np.array(Tl),np.array(SA),abscissa='T',ordinate=ord,xi=xi)
	rs.setLabel('EC8 Type '+str(stype)+' GT '+str(gtype)+' '+str(xi*100)+'%')
	return rs

@config.set_module('qtools')
def dec8_2017(SRP, sa_unit='g', xi=0.05, inf=10, gtype='A', q=1.0, option='HE'):
	"""
	Create a response spectrum in accordance with the draft revision of BS EN
	1998-1 (2017). This function will generate a spectrum in the range
	`T` = [0.01;10] sec or `f` = [0.1;100] Hz.

	Parameters
	----------
	SRP : tuple of floats with 2 elements
		The two spectral parameters :math:`S_{sRP}` and :math:`S_{1RP}`.
	sa_unit : {'g', 'm/s**2'}, optional
		Unit of the spectral parameters. Default value 'g'.
	xi : float, optional
		Damping ratio. Default value 0.05.
	inf : integer, optional
		Number of points in each interval resulting in a total of `5*inf`
		points. Default value 10.
	gtype : {'A', 'B', 'C', 'D', 'E'}, optional
		Ground type. Default value 'A'.
	q : float, optional
		Behaviour factor.
	option : {'HE', 'HD', 'VE', 'VD'}, optional
		Type of spectrum: Horizontal / Vertical, Elastic / Design.
		The default value, 'HE', is the horizontal elastic spectrum.

	Returns
	-------
	rs : an instance of class ResponseSpectrum
	"""

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
		config.vprint('Warning: option HD has not been implemented yet.')
	elif option=='VE':
		config.vprint('Warning: option VE has not been implemented yet.')
	elif option=='VD':
		config.vprint('Warning: option VD has not been implemented yet.')
	else:
		raise ValueError('Unsupported option in call to dec8_2017: option = {}'.format(option))

	if sa_unit == 'g':
		ord = 'sag'
	else:
		ord = 'sa'

	rs = ResponseSpectrum(np.array(Tl),np.array(SA),abscissa='T',ordinate=ord,xi=xi)
	rs.setLabel('DEC8 GT '+str(gtype)+' '+str(xi*100)+'%')
	return rs

@config.set_module('qtools')
def ieee693_2005(option='moderate', inf=10, xi=0.02):
	"""
 	Generate the required response spectrum in accordance with
	IEEE Std 693-2005.

	Parameters
	----------
	option : {'moderate', 'high'}, optional,
		Specify qualification level. Default 'moderate'.
	inf : int, optional
		Number of points in each interval resulting in a total of `4*inf`
		points. Default value 10.
	xi : float, optional
		Damping ratio. Default value 0.02.

	Returns
	-------
	rs : an instance of class ResponseSpectrum
	"""

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

	rs = ResponseSpectrum(np.array(fl), np.array(SA), abscissa='f' ,ordinate='sag', xi=xi)
	rs.setLabel('IEEE 693 RRS '+str(xi*100)+'%')
	return rs

@config.set_module('qtools')
def pml(pga, pga_unit='g', xi=0.05, inf=10, gtype='hard'):
	"""
	Create a PML design response spectrum. This function will generate a
	spectrum in the range `T` = [0.01;10] sec or `f` = [0.1;100] Hz.

	Parameters
	----------
	pga : float
		The peak ground acceleration.
	pga_unit : {'g', 'm/s**2'}, optional
		Unit of the pga value. Default value 'g'.
	xi : float, optional
		Damping ratio. Default value 0.05.
	inf : integer, optional
		Number of points in each interval resulting in a total of `5*inf`
		points. Default value 10.
	gtype : {'hard', 'medium', 'soft'}, optional
		Ground type. Default value 'hard'.

	Returns
	-------
	rs : an instance of class ResponseSpectrum
	"""

	if pga_unit == 'g':
		ord = 'sag'
		g = 9.81
	else:
		ord = 'sa'
		g = 1.0

	gtype = gtype.lower()
	if gtype not in ('hard','medium','soft'):
		raise ValueError('Unsupported ground type in call to pml function.')

	params = {'hard' : [0.038, 6.203, 4.89, 1.15, 3.64, 0.78, 3.21, 0.26, 12.0, 40.0],
			  'medium' : [0.056, 3.465, 5.34, 1.19, 4.0, 0.85, 3.03, 0.39, 10.0, 33.0],
			  'soft' : [0.067, 3.335, 5.54, 1.26, 4.28, 0.91, 3.48, 0.46, 8.0, 33.0]}

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

	rs = ResponseSpectrum(np.array(fl), np.array(SA), abscissa='f', ordinate=ord, xi=xi)
	rs.setLabel('PML '+str(gtype)+' '+str(xi*100)+'%')
	return rs

@config.set_module('qtools')
def eur_vhr(pga, pga_unit='g', inf=10):
	"""
	Create a response spectrum in accordance with the draft EUR Very Hard Rock
	spectrum (model 1) proposed by `Dalguer & Renault (2017)`_.
	This function will generate a spectrum in the range
	`T` = [0.005;10] sec or `f` = [0.1;200] Hz. The spectrum will have 5%
	damping.

	Parameters
	----------
	pga : float
		Peak gropund acceleration
	pga_unit : {'g', 'm/s**2'}, optional
		Unit of the pga value. Default value 'g'.
	inf : integer, optional
		Number of points in each interval resulting in a total of `4*inf`
		points. Default value 10.

	Returns
	-------
	rs : an instance of class ResponseSpectrum

	References
	----------

	.. _Dalguer & Renault (2017):

	Dalguer, L.A., & Renault, P.L.A., 2017: "Design response spectra for very
	hard rock based in Swiss site-specific seismic hazard model", *16th World
	Conference on Earthquake Engineering*, Santiago Chile, January 9th to
	13th.
	"""

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

	if pga_unit == 'g':
		ord = 'sag'
	else:
		ord = 'sa'

	rs = ResponseSpectrum(np.array(Tl), np.array(SA), abscissa='T', ordinate=ord, xi=0.05)
	rs.setLabel('EUR VHR Model 1 (5%)')
	return rs