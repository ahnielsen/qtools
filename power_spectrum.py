"""
Package: Qtools
Module: power_spectrum
(C) 2020-2021 Andreas H. Nielsen
See README.md for further details.
"""

import numpy as np
from math import pi, sqrt, ceil
from qtools import time_history
from qtools.config import set_module, Info, version

class Spectrum:
	"""Base class for Fourier and power spectra.

	Attributes
	----------
	f : 1D NumPy array
		Frequency in Hz.
	unit : str, optional
		Unit of the spectral ordinates (e.g. 'g**2*s'). Used only for plotting.
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

@set_module('qtools')
class PowerSpectrum(Spectrum):
	"""
	This class provides attributes and methods for power spectra.

	Parameters
	----------
	f : 1D NumPy array
		Frequency in Hz.
	Wf : 1D NumPy array
		Single-sided power spectral density as a function of frequency (Hz).
	unit : str, optional
		Unit of the spectral ordinates (e.g. 'g**2*s'). Used only for plotting.
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
	w : 1D NumPy array.
		Circular frequency in rad/s.
	Wf : 1D NumPy array
		Single-sided power spectral density as a function of frequency (Hz).
	Gw : 1D NumPy array
		Single-sided power spectral density as a function of circular frequency
		(rad/s).
	unit : str
		Unit of the spectral ordinates (e.g. 'g**2*s'). Used only for plotting.
	label : str
		A user-defined label (inherited from the time history when the calcps()
		method is used).
	fmt : str
		The line format to be used when plotting this spectrum (see notes on
		`fmt` parameter in the `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_).

	Notes
	-----
	In Qtools, the term `power spectrum` is synonymous with `power spectral
	density`. All instances of this class contain power spectral densities.
	
	The different frequency measures are related by :math:`\\omega = 2 \\pi f`
	where :math:`f` is the frequency in Hz and :math:`\\omega` is the circular
	frequency in rad/s. The different spectral densities are related by:

	.. math::

		W(f) = 2 \\pi G(\\omega)


	"""
	def __init__(self, f, Wf, unit=None, label='_nolegend_', fmt=''):

		Spectrum.__init__(self, f, unit=unit, label=label, fmt=fmt)
		self.Wf = Wf
		self.w = 2*pi*f
		self.Gw = Wf/(2*pi)

	def moment(self, n):
		"""
		Calculates an approximation to the n'th moment of the
		power spectrum defined as:

		.. math::

			\\lambda_{n} = \\int_{0}^{\\infty} \\omega^{n} G(\\omega)d\\omega

		where :math:`G(\\omega)` is the one-sided power spectral density
		function.

		Parameters
		----------
		n : int
			The order of the moment.

		Returns
		-------
		A float approximating the n'th moment.
		"""

		return np.trapz(self.w**n*self.Gw, x=self.w)

	def moments(self, *ns):
		"""Calculate multiple moments in one go. See :meth:`moment` for
		further information.

		Parameters
		----------
		\*ns : int
			One or more integers.

		Returns
		-------
		List
			A list of moments whose length is equal to the number of arguments
			provided to the method.
		"""

		return [self.moment(n) for n in ns]

	def rms(self):
		"""
		This method calculates the root mean square from the power spectrum.
		"""

		return sqrt(self.moment(0))



@set_module('qtools')
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
		Unit of the spectrum (e.g. 'g'). Used only for plotting.
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
		Unit of the spectrum (e.g. 'g'). Used for plotting.
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

@set_module('qtools')
def calcps(th):
	"""
	Calculates an estimate of the power spectrum corresponding to a time
	history considered as a sample from a stationary and ergodic process.

	Parameters
	----------
	th : an instance of class :class:`.TimeHistory`
		The input time history.

	Returns
	-------
	ps : an instance of :class:`.PowerSpectrum`

	Notes
	-----
	The smoothed power spectrum is calculated in accordance with a
	procedure described by `Newland (1993)`_.

	References
	----------
	.. _Newland (1993):

	Newland, D.E., 1993: *An Introduction to Random Vibrations, Spectral &
	Wavelet Analysis*, 3rd Ed., Pearson Education Ltd.
	"""
	# Initial checks etc.
	lbl = th.label if th.label != '_nolegend_' else ''
	Info.note('Computing power spectum for time history {}'.format(lbl))
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
		Info.warn('The time history does not represent acceleration, '
			'velocity or displacement.')

	# Ensure time history has zero mean
	th.zero_mean()

	# Determine required size of array. This will pad the input th with zeros
	# up to first 2**p integer that is greater than th.ndat)
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
	#Info.note('Smoothing parameter ns = ',ns)
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

	Info.end()

	# Create and return the spectrum
	ps = PowerSpectrum(f, Wf, unit=unit)
	ps.setLabel(th.label)
	return ps

@set_module('qtools')
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
	fs : an instance of :class:`.FourierSpectrum`

	Notes
	-----
	The original time history is recovered exactly (within numerical precision)
	by using :func:`.calcth` provided ``M = th.ndat``.

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
	Info.note('Converting time history {} into a Fourier'
			   ' spectrum.'.format(th.label))
	if L == 0:
		Info.note('The original length of the time history is used with'
				' {} data points'.format(N))
	elif L > 0:
		Info.note('Added {} zeros ({:5.2f} sec) to the end of the time'
				' history.'.format(L,L*th.dt))
	else:
		Info.warn('Removed {} data points from the end of the'
				' time history.'.format(L))

	# Create and return the spectrum
	Info.end()
	fs = FourierSpectrum(f, X, N, th.dt, L)
	fs.setLabel(th.label)
	return fs

@set_module('qtools')
def transfer(ps, fn, xi, tfun=0):
	"""
	Transfers a spectrum from base to in-structure for a SDOF system.

	Parameters
	----------
	ps : a instance of class PowerSpectrum or FourierSpectrum
		Input spectum.
	fn : float
		Center frequency of the transfer function.
	xi : float
		Damping ratio of the transfer function.
	tfun : int, optional
		Type of transfer function: see notes below. The default is 0.

	Returns
	-------
	An instance of class PowerSpectrum or FourierSpectrum
		Output spectrum.

	Notes
	-----
	The following SDOF transfer functions are supported:
	
	0. Input: total acceleration --> output: relative displacement
	1. Input: total acceleration --> output: total acceleration
	
	All transfer functions assume viscous damping.
	"""

	wn = 2*pi*fn
	w = 2*pi*ps.f

	if isinstance(ps, PowerSpectrum):
		if tfun==0:
			if ps.unit=='m**2/s**3':
				unit = 'm**2*s'
			else:
				Info.warn('Could not recognise unit in transfer function.')
				unit = None
			H2 = 1/((wn**2 - w**2)**2 + (2*xi*wn*w)**2)
		elif tfun==1:
			unit = ps.unit
			H2 = (wn**4 + (2*xi*wn*w)**2)/((wn**2 - w**2)**2 + (2*xi*wn*w)**2)
		
		Wy = H2*ps.Wf
		return PowerSpectrum(ps.f, Wy, unit=unit)

	if isinstance(ps, FourierSpectrum):
		if tfun==0:
			if ps.unit=='m/s**2':
				unit = 'm'
			else:
				unit = None
			H = -1/(wn**2 - w**2 + 2*1j*xi*wn*w)
		elif tfun==1:
			H = -(2*1j*xi*wn*w + wn**2)/(wn**2 - w**2 + 2*1j*xi*wn*w)
		
		Y = H*ps.X
		return FourierSpectrum(ps.f, Y, ps.N, ps.dt, L=ps.L, unit=unit)

@set_module('qtools')
def savesp(ps, fname, abscissa='f', fmt='%.18e', delimiter=' ',
		 newline='\n', header='_default_', footer='', comments='# '):
	"""Save the spectrum to text file. This function uses
	:func:`numpy.savetxt` to perform the output operation. See
	documentation for :func:`numpy.savetxt` to understand the full
	functionality.

	Parameters
	----------
	ps : an instance of class FourierSpectrum or PowerSpectrum
		The spectrum to be saved to file.
	fname : str
		Filename or file handle. See :func:`numpy.savetxt`.
	abscissa : str
		If set equal to 'w', then the abscissa will be circular frequency
		(in rad/s), otherwise it will be linear frequency (in Hz).
		This parameter will be ignored when `ps` is a Fourier spectrum.
		For consistency, when ``abscissa == 'w'``, the ordinate will be divided
		by :math:`2\\pi`. Default 'f'.
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

	if isinstance(ps, FourierSpectrum):
		head0 = 'Fourier '
		head2 = 'f [Hz], X [{}]'.format(ps.unit)
		out = np.array([ps.f,ps.X])
	elif isinstance(ps, PowerSpectrum):
		head0 = 'Power '
		if abscissa=='f':
			head2 = 'f [Hz], Wf [{}]'.format(ps.unit)
			out = np.array([ps.f, ps.Wf])
		elif abscissa=='w':
			head2 = 'w [rad/s], Gw [{}]'.format(ps.unit)
			out = np.array([2*pi*ps.f, ps.Sw/(2*pi)])
	else:
		raise TypeError('Wrong type of object supplied as first argument.')

	if header=='_default_':
		lbl = ps.label+' ' if ps.label != '_nolegend_' else ''
		head1 = head0+'spectrum {}computed by Qtools v. {}\n'.format(lbl,version)
		header = head1+head2

	np.savetxt(fname, out.T, fmt=fmt, delimiter=delimiter, newline=newline,
				   header=header, footer=footer, comments=comments)


@set_module('qtools')
def kanai_tajimi(we, wg, xig, G0=1, wc=None, xic=1.0, unit='m**2/s**3'):
	"""
	Computes the Kanai-Tajimi power spectrum.

	Parameters
	----------
	we : NumPy 1D array
		Exicitation frequencies in rad/s.
	wg : float
		Ground frequency. Typical values are between 15 rad/s (soft ground)
		and 40 rad/s (hard ground).
	xig : float
		Ground damping. Typical values are between 0.3 and 0.6.
	G0 : float, optional
		Factor used to scale the spectrum. The default is 1.
	wc : float, optional
		Ground frequency in the Clough & Penzien correction term (see notes
		below). If not specified, the correction term is not applied.
	xic : float, optional
		Ground damping in the Clough & Penzien correction term (see notes
		below).
	unit : str, optional
		Unit of the power spectrum. The default value ('m**2/s**3') is
		appropriate for an acceleration spectrum.

	Returns
	-------
	ps : an instance of :class:`.PowerSpectrum`

	Notes
	-----
	This function returns the Kanai-Tajimi power spectrum (see e.g.
	`Kramer (1996)`_). Optionally, the correction proposed by `Clough &
	Penzien (2003)`_ may be applied. The correction attenuates the spectral
	density at low frequencies and ensures that `G(0) = 0`. The parameters
	`wc` and `xic` (which Clough & Penzien denote :math:`\\omega_2` and
	:math:`\\xi_2`,	respectively) must be set appropriately to produce the
	desired attenuation. In particular, `wc` should be less than `wg`.

	References
	----------

	.. _Kramer (1996):

	Kramer, S.L., 1996: *Geotechnical Earthquake Engineering*, Prentice-Hall.

	.. _Clough & Penzien (2003):

	Clough, R.W., & Penzien, J., 2003: *Dynamics of Structures*, 3rd Edition,
	Computers & Structures, Inc.
	"""

	H1 = (1 + 2*1j*xig*we/wg)/(1 - (we/wg)**2 + 2*1j*xig*we/wg)

	if isinstance(wc, float):
		H = H1*(we/wc)**2/(1 - (we/wc)**2 + 2*1j*xic*we/wc)
		label = 'Kanai-Tajimi (Clough-Penzien)'
	else:
		H = H1
		label = 'Kanai-Tajimi'

	ps = PowerSpectrum(we/(2*pi), G0*np.abs(H)**2, unit=unit, label=label)
	return ps