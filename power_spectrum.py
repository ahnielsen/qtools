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

@config.set_module('qtools')
class PowerSpectrum:
	"""
	This class provides methods for calculation of Fourier and power spectra.
	``PowerSpectrum`` is a general class for Fourier spectra and power spectra.

	Parameters
	----------
	f : 1D NumPy array
		Frequency in Hz
	X : 1D NumPy array
		Fourier amplitudes
	Sk : 1D NumPy array
		Double-sided power spectrum (before smoothing) as a function of
		circular frequency (rad/s)
	Sw : 1D Numpy array
		Double-sided power spectrum (after smoothing) as a function of circular
		frequency (rad/s)
	unit : dict, optional
		The `unit` dictionary has two keys:

		* 'X' : unit of the Fourier spectrum (default 'g')
		* 'S' : unit of the power spectrum (default 'g**2*s')

	label : str, optional
		A user-defined label (inherited from the time history when the calcps()
		method is used)
	fmt : str, optional
		The line format to be used when plotting this spectrum (see notes on
		`fmt` parameter in the `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_)

	Attributes
	----------
	f : 1D NumPy array
		Frequency in Hz.
	X : 1D NumPy array
		Fourier amplitudes.
	Sk : 1D NumPy array
		Double-sided power spectrum (before smoothing) as a function of
		circular frequency (rad/s).
	Sw : 1D Numpy array
		Double-sided power spectrum (after smoothing) as a function of circular
		frequency (rad/s).
	Wf : 1D NumPy array
		Single-sided power spectrum (after smoothing) as a function of
		frequency (Hz). This is computed as :math:`W_f = 4\pi S_w`.
	unit : dict
		The `unit` dictionary has two keys:

		* 'X' : unit of the Fourier spectrum (default 'g').
		* 'S' : unit of the power spectrum (default 'g**2*s').

	label : str
		A user-defined label (inherited from the time history when the calcps()
		method is used).
	fmt : str
		The line format to be used when plotting this spectrum (see notes on
		`fmt` parameter in the `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_).

	"""
	def __init__(self, f, X, Sk, Sw, unit={'X': 'g', 'S': 'g**2*s'}, label='_nolegend_', fmt='_default_'):

		self.f = f
		self.X = X
		self.Sk = Sk
		self.Sw = Sw
		self.Wf = 4*pi*self.Sw
		self.unit = unit
		self.label = label
		self.fmt = fmt

	def setLabel(self, label):
		"""Sets the label of the response spectrum."""
		self.label = str(label)

	def setLineFormat(self, fmt):
		"""Sets the `fmt` attribute equal to the supplied argument."""
		self.fmt = str(fmt)

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

	def saveps(self, fname, ordinate='Wf', fmt='%.18e', delimiter=' ',
			 newline='\n', header='_default_', footer='', comments='# '):
		"""Save the spectrum to text file. This function uses
		:func:`numpy.savetxt` to perform the output operation. See
		documentation for :func:`numpy.savetxt` to understand the full
		functionality.

		Parameters
		----------
		fname : str
			Filename or file handle. See :func:`numpy.savetxt`.
		ordinate : {'Wf', 'Sw', 'Sk', 'X', 'all'}, optional
			Output power spectrum ('Wf', 'Sw' or 'Sk'), Fourier spectrum ('X'),
			or all quantities ('all'). Default 'Wf'.
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
			head1 = 'Power spectrum computed by Qtools v. {}\n'.format(config.version)

		if ordinate=='Wf':
			head2 = 'f [Hz], Wf [{}]'.format(self.unit['S'])
			out = np.array([self.f,self.Wf])
		elif ordinate=='Sw':
			head2 = 'f [Hz], Sw [{}]'.format(self.unit['S'])
			out = np.array([self.f,self.Sw])
		elif ordinate=='Sk':
			head2 = 'f [Hz], Sk [{}]'.format(self.unit['S'])
			out = np.array([self.f,self.Sk])
		elif ordinate=='X':
			head2 = 'f [Hz], X [{}]'.format(self.unit['X'])
			out = np.array([self.f,self.X])
		elif ordinate=='all':
			head2 = 'f [Hz], X [{0}], Sk [{1}], Sw [{1}], Wf [{1}]'.format(self.unit['X'],self.unit['S'])
			out = np.array([self.f,self.X,self.Sk,self.Sw,self.Wf])
		else:
			config.vprint('WARNING: could not recognise the parameter ordinate = {}'.format(ordinate))
			config.vprint('No data has been save to file')
			return

		if header=='_default_':
			header = head1+head2

		np.savetxt(fname,out.T,fmt=fmt,delimiter=delimiter,newline=newline,
					   header=header,footer=footer,comments=comments)

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
	procedure described in [1]_.

	References
	----------
	.. [1] Newland, D.E., *An Introduction to Random Vibrations, Spectral &
	   Wavelet Analysis*, 3rd Ed., Pearson Education Ltd., 1993.
	"""
	# Initial checks etc.
	config.vprint('Computing power spectum for time history {}'.format(th.label))
	if type(th) != time_history.TimeHistory:
		raise TypeError('The th parameter must be specified as an instance of class TimeHistory')
	if not th.dt_fixed:
		raise ValueError('To compute a power spectrum a time history must be defined with fixed time step')

	# Define g
	if th.ordinate == 'a':
		g = 9.81
	else:
		g = 1.

	# Ensure time history has zero mean
	th.zero_mean()

	# Determine required size of array
	N = th.ndat
	p = 1
	while N-2**p > 0:
		p += 1
	M = 2**p
	config.vprint('Size of original time history = ',N)
	config.vprint('Size of zero-padded time history = ',M)
	# Execute FFT
	Xc = 1/M*np.fft.rfft(th.data,n=M)
	K = np.size(Xc)
	# Fourier amplitude
	X = np.absolute(Xc)/g
	TL = th.dt*M
	Td = th.time[-1]
	#config.vprint('Duration of original time history =',Td)
	f = np.fft.rfftfreq(M,d=th.dt)
	# Unsmoothed spectrum
	Sk = M/N*TL/(2*pi)*X**2
	# Smoothed spectrum
	#
	# TO DO: Investigate whether smoothing should be done over an interval
	# that is proportional to frequency (such that smoothing covers a wider
	# frequency band at higher frequencies)
	#
	# Assume the required bandwidth is 0.5Hz
	Be = 0.5
	ns = ceil(M*Be*Td/(2*N)-0.5)
	config.vprint('Smoothing parameter ns = ',ns)
	# Calculate smoothed spectrum
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
	# Units
	unit = {}
	if th.ordinate == 'd':
		unit['X'] = 'm'
		unit['S'] = 'm**2*s'
	elif th.ordinate == 'v':
		unit['X'] = 'm/s'
		unit['S'] = '(m/s)**2*s'
	elif th.ordinate == 'a':
		unit['X'] = 'g'
		unit['S'] = 'g**2*s'
	# Create and return the spectrum
	ps = PowerSpectrum(f, X, Sk, Sw, unit)
	ps.setLabel(th.label)
	config.vprint('------------------------------------------')
	return ps

@config.set_module('qtools')
def rs2ps(rs,Mw,R,region,Td=None,method='V75'):
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
		config.vprint('WARNING: Remember to protect the main module with an if __name__ == \'__main__\': statement')
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