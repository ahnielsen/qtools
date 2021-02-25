"""
Package: Qtools
Module: power_spectrum
(C) 2020-2021 Andreas H. Nielsen
See README.md for further details.
"""

import numpy as np
from scipy.integrate import romb
from math import pi, sqrt, ceil
from . import time_history, response_spectrum, config
try:
	import pyrvt
except ModuleNotFoundError:
	config.vprint('WARNING: package PyRVT not found. Calling function rs2ps '
			   'will have no effect.')
	PYRVT_PRESENT = False
else:
	PYRVT_PRESENT = True

class Spectrum:
	"""Base class for Fourier and power spectra.

	Attributes
	----------
	f : 1D NumPy array
		Frequency in Hz.
	unit : str, optional
		Unit of the spectral ordinates (e.g. `g**2*s`). Used only for plotting.
	label : str
		A user-defined label.
	fmt : str
		The line format to be used when plotting this spectrum (see notes on
		`fmt` parameter in the `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_).
	"""
	def __init__(self, f, unit=None, label='_nolegend_', fmt=''):

		self.f = f
		self.unit = unit
		self.label = label
		self.fmt = fmt

	def setLabel(self, label):
		"""Sets the label of the spectrum."""
		self.label = str(label)

	def setUnit(self, unit):
		"""Sets the unit of the spectral ordinates."""
		self.label = str(unit)

	def setLineFormat(self, fmt):
		"""Sets the `fmt` attribute equal to the supplied argument."""
		self.fmt = str(fmt)

@config.set_module('qtools')
class PowerSpectrum(Spectrum):
	"""
	This class provides attributes and methods for power spectra.

	Parameters
	----------
	f : 1D NumPy array
		Frequency in Hz.
	Wf : 1D Numpy array
		Single-sided power spectrum as a function of frequency (Hz).
	unit : str, optional
		Unit of the spectral ordinates (e.g. `g**2*s`). Used only for plotting.
	label : str, optional
		A user-defined label (inherited from the time history when the
		:func:`.calcps` function is used).
	fmt : str, optional
		The line format to be used when plotting this spectrum (see notes on
		`fmt` parameter in the `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_)

	Attributes
	----------
	f : 1D NumPy array
		Frequency in Hz.
	Wf : 1D NumPy array
		Single-sided power spectrum as a function of frequency (Hz).
	unit : str
		Unit of the spectral ordinates (e.g. `g**2*s`). Used only for plotting.
	label : str
		A user-defined label (inherited from the time history when the calcps()
		method is used).
	fmt : str
		The line format to be used when plotting this spectrum (see notes on
		`fmt` parameter in the `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_).

	"""
	def __init__(self, f, Wf, unit=None, label='_nolegend_', fmt=''):

		Spectrum.__init__(self, f, unit=unit, label=label, fmt=fmt)
		self.Wf = Wf

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

@config.set_module('qtools')
class FourierSpectrum(Spectrum):
	"""
	This class provides attributes and methods for Fourier spectra.

	Parameters
	----------
	f : 1D NumPy array
		Frequency in Hz.
	X : 1D NumPy array
		Complex Fourier amplitudes.
	dt : float
		Time step.
	unit : str, optional
		Unit of the spectrum (e.g. `g`). Used only for plotting.
	label : str, optional
		A user-defined label (inherited from the time history when the
		:func:`.calcfs` function is used)
	fmt : str, optional
		The line format to be used when plotting this spectrum (see notes on
		`fmt` parameter in the `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_)

	Attributes
	----------
	f : 1D NumPy array
		Frequency in Hz.
	X : 1D NumPy array
		Complex Fourier amplitudes.
	dt : float
		Time step.
	unit : str
		Unit of the spectrum (e.g. `g`). Used for plotting.
	label : str
		A user-defined label (inherited from the time history when the calcps()
		method is used).
	fmt : str
		The line format to be used when plotting this spectrum (see notes on
		`fmt` parameter in the `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_).

	Notes
	-----
	The absolute amplitude `|X|` of the spectrum is used for plotting.
	"""
	def __init__(self, f, X, N, dt, L=0, unit=None, label='_nolegend_', fmt=''):

		Spectrum.__init__(self, f, unit=unit, label=label, fmt=fmt)
		self.X = X
		self.N = N
		self.L = L
		self.dt = dt

	def abs(self):
		"""Returns the absolute amplitude as a NumPy array."""
		return np.absolute(self.X)

@config.set_module('qtools')
def calcps(th):
	"""
	Calculates a power spectrum from a time history.

	Parameters
	----------
	th : an instance of class :class:`.TimeHistory`
		The input time history.

	Returns
	-------
	ps : an instance of class :class:`.PowerSpectrum`

	Notes
	-----
	The smoothed spectrum (`Sw` and `Wf`) is calculated in accordance with a
	procedure described by `Newland (1993)`_.

	References
	----------
	.. _Newland (1993):

	Newland, D.E., 1993: *An Introduction to Random Vibrations, Spectral &
	Wavelet Analysis*, 3rd Ed., Pearson Education Ltd.
	"""
	# Initial checks etc.
	config.vprint('Computing power spectum for time history {}'.format(th.label))
	if type(th) != time_history.TimeHistory:
		raise TypeError('The th parameter must be specified as an instance of'
				  ' class TimeHistory')
	if not th.dt_fixed:
		raise ValueError('To compute a power spectrum, a time history must be'
				   ' defined with fixed time step')
	if th.ordinate == 'a':
		unit = 'm**2/s**3'
	elif th.ordinate == 'v':
		unit = 'm**2/s'
	elif th.ordinate == 'd':
		unit = 'm**2*s'
	else:
		config.vprint('WARNING: the time history does not represent '
				'acceleration, velocity or displacement.')

	# Ensure time history has zero mean
	th.zero_mean()

	# Determine required size of array
	N = th.ndat
	p = 1
	while N-2**p > 0:
		p += 1
	M = 2**p

	# Execute FFT
	Xc = 1/M*np.fft.rfft(th.data, n=M)
	K = np.size(Xc)
	# Fourier amplitude
	X = np.absolute(Xc)

	# Durations and frequencies
	# NB: the period T of the signal is the duration of the time history plus
	# a time step
	TL = th.dt*M
	T = th.Td+th.dt
	f = np.fft.rfftfreq(M, d=th.dt)

	# Unsmoothed spectrum
	Sk = M/N*TL/(2*pi)*X**2

	# Calculate smoothed spectrum
	#
	# TO DO: Investigate whether smoothing should be done over an interval
	# that is proportional to frequency (such that smoothing covers a wider
	# frequency band at higher frequencies)
	#
	# Assume the required bandwidth is 0.5Hz
	Be = 0.5
	ns = ceil(M*Be*T/(2*N)-0.5)
	config.vprint('Smoothing parameter ns = ',ns)
	Sw = np.zeros(K)
	for k in range(K):
		sswkm = 0.0
		for m in range(-ns,ns+1):
			i = k+m
			if i < 0:
				sswkm += Sk[-i]
			elif i >= 0 and i < K-1:
				sswkm += Sk[i]
		Sw[k] = 1/(2*ns+1)*sswkm
		
	# Convert to single-sided spectrum
	Wf = 4*pi*Sw

	# Create and return the spectrum
	ps = PowerSpectrum(f, Wf, unit=unit)
	ps.setLabel(th.label)
	return ps

@config.set_module('qtools')
def calcfs(th, M=None):
	"""
	Calculates a Fourier spectrum from a time history.

	Parameters
	----------
	th : an instance of class :class:`.TimeHistory`
		The input time history.
	M : int
		Number of points in the input time history to use. If `M` is smaller
		than the length of the input (`th.ndat`), the input is cropped. If it
		is larger, the input is padded with zeros. If `M` is not given, the
		length of the input along the axis specified by axis is used.

	Returns
	-------
	fs : an instance of class :class:`.FourierSpectrum`

	Notes
	-----
	The original time history is recovered exactly (within numerical precision)
	by using :func:`.calcth` provided `M = th.ndat`.

	"""
	# Initial checks etc.
	if type(th) != time_history.TimeHistory:
		raise TypeError('The th parameter must be specified as an instance of'
				  ' class TimeHistory')
	if not th.dt_fixed:
		raise ValueError('To compute a Fourier spectrum, a time history must'
				   ' be defined with fixed time step')

	# Determine required size of array
	if M is None:
		N = th.ndat
	else:
		N = M
	# L is the number of zeros added to the time history
	L = N - th.ndat
	# Execute FFT
	X = 1/N*np.fft.rfft(th.data, n=M)
	f = np.fft.rfftfreq(N, d=th.dt)

	# Output
	config.vprint('Converting time history {} into a Fourier'
			   ' spectrum.'.format(th.label))
	if L == 0:
		config.vprint('The original length of the time history is used with'
				' {} data points'.format(N))
	elif L > 0:
		config.vprint('Added {} zeros ({:5.2f} sec) to the end of the time'
				' history.'.format(L,L*th.dt))
	else:
		config.vprint('WARNING: Removed {} data points from the end of the'
				' time history.'.format(L))
	config.vprint('------------------------------------------')

	# Create and return the spectrum
	fs = FourierSpectrum(f, X, N, th.dt, L)
	fs.setLabel(th.label)
	return fs

@config.set_module('qtools')
def savesp(sp, fname, ordinate, fmt='%.18e', delimiter=' ',
		 newline='\n', header='_default_', footer='', comments='# '):
	"""Save the spectrum to text file. This function uses
	:func:`numpy.savetxt` to perform the output operation. See
	documentation for :func:`numpy.savetxt` to understand the full
	functionality.

	Parameters
	----------
	sp : an instance of class Fourier Spectrum or Power Spectrum
		The spectrum to be saved to file.
	fname : str
		Filename or file handle. See :func:`numpy.savetxt`.
	ordinate : str
		Set this equal to the attribute that is to be written.
		If ``ordinate = 'Sk'`` or ``ordinate = 'Sw'``, then the abscissa will
		be circular frequency (in rad/s), otherwise it will be linear
		frequency (in Hz).
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
	if type(sp) == FourierSpectrum:
		head0 = 'Fourier '
		head2 = 'f [Hz], X [{}]'.format(sp.unit)
		out = np.array([sp.f,sp.X])
	elif type(sp) == PowerSpectrum:
		head0 = 'Power '
		if ordinate=='Wf':
			head2 = 'f [Hz], Wf [{}]'.format(sp.unit)
			out = np.array([sp.f,sp.Wf])
		elif ordinate=='Sw':
			head2 = 'w [rad/s], Sw [{}]'.format(sp.unit)
			out = np.array([2*pi*sp.f,sp.Sw])
		elif ordinate=='Sk':
			head2 = 'w [rad/s], Sk [{}]'.format(sp.unit)
			out = np.array([2*pi*sp.f,sp.Sk])
		else:
			config.vprint('WARNING: could not recognise the parameter'
				 ' ordinate = {}'.format(ordinate))
			config.vprint('No data has been save to file')
		return
	else:
		raise TypeError('Wrong type of object supplied as first argument.')

	if header=='_default_':
		head1 = head0+'spectrum computed by Qtools v. {}\n'.format(config.version)
		header = head1+head2

	np.savetxt(fname,out.T,fmt=fmt,delimiter=delimiter,newline=newline,
				   header=header,footer=footer,comments=comments)

@config.set_module('qtools')
def rs2ps(rs, Mw, R, region, Td=None, method='V75'):
	"""Converts a response spectrum into a power spectrum.

	Parameters
	----------
	rs : an instance of class :class:`.ResponseSpectrum`
		The response spectrum that is to be converted.
	Mw : float
		Earthquake magnitude (see notes below).
	R : float
		Earthquake distance in km (see notes below).
	region : str
		Region (see notes below).
	Td : float, optional
		Duration (see notes below).
	method : str, optional
		Method (see notes below).

	Returns
	-------
	tuple
		A tuple with two objects:

		0. `ps` - an instance of class :class:`.TimeHistory`.
		1. `rsc` - an instance of class :class:`.ResponseSpectrum`. This
		   spectrum contains the acceleration back-calculated from `ps`. The
		   returned spectrum `rsc` will differ slightly from the input spectrum
		   `rs`.

	Notes
	-----
	This function passes the parameters to
	:func:`pyrvt.tools.calc_compatible_spectra`. For the meaning of parameters,
	see documentation for `PyRVT <https://pyrvt.readthedocs.io/>`_.
	If PyRVT is not installed on the system, the function will do
	nothing. To use the function, the main module must be protected by an
	``if __name__ == '__main__':`` statement.
	"""

	if PYRVT_PRESENT:
		config.vprint('WARNING: Remember to protect the main module with'
				' an if __name__ == \'__main__\': statement')
		e = {}
		e['psa'] = rs.sa
		e['magnitude'] = Mw
		e['distance'] = R
		e['region'] = region
		e['duration'] = Td
		freqs = pyrvt.tools.calc_compatible_spectra(method, rs.T, [e], damping=rs.xi)

		Td = e['duration']
		config.vprint('The duration is = {}'.format(Td))
		X = e['fa']/Td
		Sw = e['fa']**2/(4*pi*Td)
		Sk = Sw
		ps = PowerSpectrum(freqs, X, Sk, Sw)
		rsc = response_spectrum.ResponseSpectrum(rs.T,e['psa_calc'],abscissa='T')

		return (ps, rsc)