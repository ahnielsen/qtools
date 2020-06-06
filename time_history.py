"""
Module qtools.time_history
Author: Andreas H Nielsen
"""

import numpy as np
from scipy.integrate import cumtrapz, trapz
from math import pi, ceil, sqrt
from qtools import config
from pathlib import Path

@config.set_module('qtools')
class TimeHistory:
	"""
	General class for time histories.

	Parameters
	----------
	time : 1D NumPy array
		Values of time for each corresponding entry in `data`.
	data : 1D NumPy array
		Time history data values (note: the lengths of `time` and `data` must
		be the same).
	ordinate : {'d', 'v', 'a'}, optional
		The physical quantity contained in data, with 'd' = displacement,
		'v' = velocity and 'a' = acceleration.
	fmt : str, optional
		The line format to be used when plotting this time history (see notes
		on the `fmt` parameter in `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_.
		With the default value, ``fmt = ''`` (an empty string), the history
		will be plotted with the default matplotlib.pyplot line styles.

	Attributes
	----------
	data : 1D NumPy array
		This array contains the ordinate values (i.e. acceleration, velocity
		or displacement).
	dt : float
		The time step (when the time step is fixed).
	dt_fixed : bool
		Is True when the time step is fixed (i.e. when
		``dt == time[i+1]-time[i]`` for all `i`.), False otherwise.
	fNyq : float
		The estimated Nyquist frequency. For fixed time step :math:`\Delta t`,
		the Nyquist frequency is given as :math:`f_{Nyq} = 1/({2\Delta t})`.
	fmt : str
		See above under Parameters.
	label : str
		A short descriptor or name for the time history.
	ndat : int
		Number of elements in `time` and `data`.
	ordinate : str
		The type of motion contained in `data`, with 'd' = displacement,
		'v' = velocity and 'a' = acceleration. Integrating or differentiating
		will change the ordinate accordingly.
	Td : float
		Duration of the time history
	time : 1D NumPy array
		This array contains time points. The array is populated even when a
		fixed time step is specified.

	Returns
	-------
	An instance of TimeHistory

	Notes
	-----
	Each instance of TimeHistory should be defined in standard SI units
	(m, m/s, |m/s2|). The :class:`.ResponseSpectrum` class assumes that
	acceleration time histories are defined in units of |m/s2|. Therefore, if
	a time history read from file is defined in g, the call should be::

		th = qt.readth('filename.txt',factor=9.81)

	Differentiation has yet to be implemented.

	Each instance of TimeHistory is iterable. For example, the following
	statement loops over all time-ordinate pairs in time history `th`::

		for t,a in th:
			# Do something with t and a.

	Each instance of TimeHistory is callable. For example, the following
	statement computes the acceleration value at time `t`::

		acc = th(t)

	The time `t` does not have to coincide with any of the discrete time points
	in the `time` attribute. However, this functionality should not be used to
	define a new time history with shorter time steps: to this end, use the
	class method :meth:`interpft`.
	"""
	def __init__(self,time,data,ordinate='a',fmt=''):

		if len(time) != len(data):
			raise IndexError('The lengths of time and data must be the same.')
		self.time = time
		self.data = data
		self.dt = time[1]-time[0]
		self.k = 0
		#self.l = 0
		self.fNyq = 1/(2*self.dt)
		self.ordinate = ordinate
		self.label = '_nolegend_'
		self.Td = time[-1]-time[0]
		self.ndat = len(time)
		self.fmt = fmt

		# Find the PGA and calculate rms and 0.5-0.75 duration
		if self.ordinate == 'a':
			self.pga = np.amax(np.fabs(self.data))

	def __call__(self,t):

		k = self.k
		if t <= self.time[0]:
			x = self.data[0]
			k = 0
		elif t >= self.time[-1]:
			x = self.data[-1]
			k = self.ndat-2
		else:
			t1 = self.time[k]
			t2 = self.time[k+1]
			while (t < t1 or t > t2):
				if t < t1:
					k -= 1
				elif t > t2:
					k += 1
				t1 = self.time[k]
				t2 = self.time[k+1]
			x1 = self.data[k]
			x2 = self.data[k+1]
			x = (x2-x1)/(t2-t1)*(t-t1)+x1
		self.k = k
		return x

	def __iter__(self):
		return zip(self.time,self.data)

#	def __next__(self):
#		if self.l < self.ndat:
#			t = self.time[self.l]
#			d = self.data[self.l]
#			self.l += 1
#		else:
#			self.l = 0
#			raise StopIteration
#		return (t,d)

	def setdt(self, dt, dt_fixed, fNyq):
		"""Set the values of `dt`, `dt_fixed` and `fNyq`."""
		self.dt = dt
		self.dt_fixed = dt_fixed
		self.fNyq = fNyq

	def setLabel(self, label):
		"""Set the `label` attribute of the time history."""
		self.label = str(label)

	def setLineFormat(self, fmt):
		"""Set the `fmt` attribute of the time history."""
		self.fmt = str(fmt)

	def writeth(self, filename, fmt='%8.4f, %8.4f'):
		"""Write the time history to text file.
		NOTE: Needs development
		"""
		a = np.transpose(np.array([self.time, self.data]))
		np.savetxt(filename,a,fmt=fmt)

	def interpft(self,k):
		"""Interpolate the time history via a discrete Fourier transform of the
		data points in the time history. The result is a time history with
		a time step ``self.dt = self.dt/k``.

		Parameters
		----------
		k : int
			Divisor on the time step. If `k` is not an integer, it will be
			converted to one.

		Notes
		-----
		The time history must have fixed time step.
		If ``self.dt_fixed is False``, no action is taken by invoking this
		method. The parameter `k` must be a positive integer. A value
		``k = 1`` results in the same time history. This method uses
		:func:`numpy.fft.rfft` and :func:`numpy.fft.irfft` to perform the
		interpolation.

		Example
		-------
		The following example shows how to use this method::

			import copy
			from math import pi
			# Create a sample time hitory
			t0, dt = np.linspace(0,3*pi,20,retstep=True)
			g0 = np.sin(t0)**2*np.cos(t0)
			th0 = qt.arrayth(g0,time=t0,fmt='o')
			# Make a copy of th0 for later use
			th1 = copy.deepcopy(th0)
			th1.setLineFormat('-')
			# Interpolate and plot the two time histories
			th1.interpft(8)
			qt.plotth(th0,th1)

		"""
		k = int(k)
		if k < 1:
			raise ValueError('The parameter k must be greater than or equal to 1.')

		if self.dt_fixed:
			n = self.ndat
			m = k*n
			cn = 1/n*np.fft.rfft(self.data)
			self.data = np.fft.irfft(m*cn, m)[0:(n-1)*k+1]
			t0 = self.time[0]; Td = self.Td
			self.time, self.dt = np.linspace(t0,t0+Td,(n-1)*k+1,retstep=True)
			self.ndat = (n-1)*k+1
			self.k = 0
			self.fNyq = 1/(2*self.dt)
		else:
			config.vprint('Warning: calling interpft() on a time history with '
				 'variable time step has no effect.')

	def smd(self, f0=0.05, f1=0.75):
		"""Compute the strong motion duration.

		Parameters
		----------
		f0 : float, optional
			Fraction of Arias intensity at the beginning of the strong motion
			duration. Default 0.05.
		f1 : float, optional
			Fraction of Arias intensity at the end of the strong motion duration.
			Default 0.75.

		Returns
		-------
		tuple
			A tuple with three floats:

			0. `smd` - the strong motion duration (``smd = t1 - t0``)
			1. `t0` - time when the Arias intensity reached `f0`.
			2. `t1` - time when the Arias intensity reached `f1`.
		"""
		if self.ordinate != 'a':
			config.vprint('WARNING: class method smd() is intended for acceleration time histories')
		Ea = cumtrapz(self.data**2,self.time,initial=0)
		t0,t1 = np.interp([f0*Ea[-1],f1*Ea[-1]],Ea,self.time)
		smd = t1-t0
		return (smd,t0,t1)

	def rms(self):
		"""Compute the root mean square.

		Notes
		-----
		This function returns an estimate of the root mean square:

		.. math:: RMS = \\sqrt{\\frac{1}{T_d} \\int_{0}^{T_d} f^2(t) dt}

		where :math:`f(t)` is the time history being operated on, and
		:math:`T_d` is the duration.
		"""
		return sqrt(trapz(self.data**2,self.time)/self.Td)

	def integrate(self):
		"""Integrate the time history with respect to time."""

		if self.ordinate=='d':
			config.vprint('WARNING: you are integrating a displacement time history.')
			self.ordinate = 'n/a'
		elif self.ordinate=='v':
			config.vprint('The time history has been integrated and now represents displacement.')
			self.ordinate = 'd'
		elif self.ordinate=='a':
			config.vprint('The time history has been integrated and now represents velocity.')
			self.ordinate = 'v'

		self.data = cumtrapz(self.data,self.time,initial=0)

	def zero_mean(self):
		"""Ensure that the time history has zero mean."""
		self.data -= np.mean(self.data)

@config.set_module('qtools')
def harmonic(dt, f=1.0, Td=None, A=1.0, ordinate='a'):
	"""
	Create a sinusoidal time history.

	Parameters
	----------
	dt : float
		The time step.
	f : float, optional
		The frequency of the time history (in Hz).
	Td : float, optional
		The duration of the time history. The default value is ``Td = 100*dt``.
	A : float, optional
		The amplitude of the time history.
	ordinate : {'d', 'v', 'a'}, optional
		The physical quality of the data generated by this function,
		with 'd' = displacement, 'v' = velocity, 'a' = acceleration.

	Returns
	-------
	th : an instance of class TimeHistory

	Notes
	-----
	If ``Td/dt`` is not a whole number, then the duration will be increased
	such that ``Td = ceil(Td/dt)*dt``.
	"""

	if Td is None:
		Td = 100*dt
	n = ceil(Td/dt)
	Tdn = n*dt
	if Tdn != Td:
		config.vprint('WARNING: the duration has been increased to {:6.2f} '
				'seconds to accommodate {} timesteps'.format(Tdn,n))
	Td = Tdn
	w = 2*pi*f
	time = np.linspace(0.,Tdn,n)
	data = A*np.sin(w*time)
	th = TimeHistory(time,data,ordinate)
	th.setdt(dt,True,1/(2*dt))

	return th

@config.set_module('qtools')
def loadth(sfile, ordinate='a', dt=-1.0, factor=1.0, delimiter=None,
		   comments='#',skiprows=0):
	"""
	Create a time history from a text file.

	Parameters
	----------
	sfile : str
		The path and name of the input file.
	ordinate : {'d', 'v', 'a'}, optional
		The physical type of the data imported from `sfile`, with
		'd' = displacement, 'v' = velocity, 'a' = acceleration.
	dt : float, optional
		The time step. A positive value forces a constant time step, in which
		case `sfile` must not contain any time values. A negative value
		(e.g. ``dt = -1.0``) signifies that the time step is specified in the
		input file.
	factor : float, optional
		This argument can be used to factor the input data. Useful for
		converting from g units to |m/s2|.
	delimiter : str, optional
		The character used to separate values. The default is whitespace.
	comments : string or sequence of strings, optional
		The characters or list of characters used to indicate the start of a
		comment. None implies no comments. The default is '#'.
	skiprows : int, optional
		Skip the first `skiprows` lines. Default 0.

	Returns
	-------
	th : an instance of class TimeHistory

	Notes
	-----
	The Nyquist frequency is calculated as:

	.. math:: f_{Nyq} = \\frac{1}{2\\Delta t_{max}}

	where :math:`\Delta t_{max}` is the longest time step in the input file
	(when the function is called with `dt` < 0) or simply equal to `dt`
	(when the function is called with `dt` > 0).

	The time step can be specified in the input file by including a comment
	with the time step value. This comment must contain 'dt =' followed by
	the value of the time step in seconds. See example A below.

	The input file can have multiple data points per line. If the time step has
	been specified through `dt` or through a comment in the input file, it is
	assumed that the data in the input file are ordinates (displacements,
	velocities, accelerations) only; if the time step has not been specified,
	it is assumed that the data are given as pairs of time and ordinate.

	Examples
	--------
	*Example A*. Fixed time step specified in a comment using white space as
	delimiter with 5 data points per line::

		# dt = 0.005
		a0 a1 a2 a3 a4
		a5 a6 a7 a8 a9
		...

	*Example B*. Time points defined directly in input file with 4 data
	points per line (resulting in 8 columns of data) using comma as delimiter::

		t0, a0, t1, a1, t2, a2, t3, a3
		t4, a4, t5, a5, t6, a6, t7, a7
		...

	"""

	if type(sfile)!=str:
		raise TypeError('The first argument to loadth() must be a string.')

	if dt > 0.0:
		dt_fixed = True
	else:
		dt_fixed = False

	if not dt_fixed:
		# Look for the time step definition in the input file
		fil = open(sfile,'r')
		for line in fil.readlines():
			if 'dt =' in line:
				met = line.partition('=')[2].lstrip()
				try:
					dt = float(met)
				except:
					pass
				else:
					dt_fixed = True
					break
		fil.close()

	# Now read the data from the specified file
	rawdata = np.loadtxt(sfile,delimiter=delimiter,comments=comments,skiprows=skiprows)
	ndat = np.size(rawdata)

	if dt_fixed:
		# Constant time step - the input file contains just ordinates
		fNyq = 1/(2*dt)
		time = np.linspace(0.0,dt*(ndat-1),num=ndat)
		data = np.reshape(rawdata,ndat)
	else:
		# The input file is defined as pairs of (time, data)
		if (np.size(rawdata[0])%2 != 0):
			# There is an odd number of entries in the first row.
			# This format is not supported.
			raise IndexError('In loadth(): when the time points are defined in '
					'the input file, an odd number of columns in the input file is not supported.')
		time = np.reshape(rawdata[:,0::2],np.size(rawdata[:,0::2]))
		data = np.reshape(rawdata[:,1::2],np.size(rawdata[:,1::2]))
		ndat //= 2
		# Find the highest frequency that can be represented by the time history
		# and check for fixed time step
		dt0 = time[1]-time[0]
		ddt = 0.0; fNyq = 1.E6
		for i in range(ndat-1):
			dt1 = time[i+1]-time[i]
			ddt = max(ddt,abs(dt1-dt0))
			fNyq = min(1/(2*dt1),fNyq)
		# Check whether time step really is fixed
		if ddt < 1E-8:
			dt_fixed = True
			dt = dt0

	# Factor input data
	if factor != 1.0:
		data *= factor

	# Create instance of TimeHistory
	th = TimeHistory(time,data,ordinate)
	th.setdt(dt,dt_fixed,fNyq)

	# Set label (removing the file path and the extension, if any)
	th.setLabel(Path(sfile).stem)

	# Output information
	config.vprint('Time history successfully read from file {}.'.format(sfile))
	if th.dt_fixed:
		config.vprint('Constant time step =',th.dt,'seconds will be used.')
	else:
		config.vprint('Variable time step will be used.')
	config.vprint('The duration is {:6.2f} seconds.'.format(th.Td))
	config.vprint('The estimated Nyquist frequency of the time history is {:6.2f} Hz.'.format(th.fNyq))
	config.vprint('There are',th.ndat,'points in the time history.')
	if ordinate == 'a':
		config.vprint('The PGA is {:6.2f} m/s**2.'.format(th.pga))
	config.vprint('------------------------------------------')

	return th

@config.set_module('qtools')
def arrayth(data, time=None, dt=-1.0, ordinate='a',fmt=''):
	"""
	Create a time history from an array.

	Parameters
	----------
	data : 1D NumPy array
		The values of the time history.
	time : 1D NumPy array, optional
		Values of time for each corresponding entry in `data`
		(note: the lengths of `time` and `data` must be the same).
	dt : float, optional
		The time step. A positive value forces a constant time step, in which
		case `time`, if given, is overwritten with new values.
		A negative value (e.g. ``dt = -1.0``) signifies that the time values
		are specified in the `time` array.
	ordinate : {'d', 'v', 'a'}, optional
		The physical type of the data in the `data` array, with
		'd' = displacement, 'v' = velocity, 'a' = acceleration.
	fmt : str, optional
		The line format to be used when plotting this time history (see class
		:class:`TimeHistory` for more information).

	Returns
	-------
	th : an instance of class TimeHistory

	Notes
	-----
	The Nyquist frequency is calculated as:

	.. math:: f_{Nyq} = \\frac{1}{2\\Delta t_{max}}

	where :math:`\Delta t_{max}` is the longest time step in the 'time' array
	(when the function is called with `dt` < 0) or simply equal to `dt`
	(when the function is called with `dt` > 0).
	"""

	# Check arguments
	if type(time) != np.ndarray and dt < 0:
		raise ValueError('Specify time or assign a positive value to dt.')
	if type(data) != np.ndarray:
		raise TypeError('The parameter data must be a NumPy array')
	if type(time) != np.ndarray and time != None:
		raise TypeError('The parameter time must be a NumPy array')
	if len(np.shape(data)) != 1:
		raise TypeError('The parameter data must be a 1D array.')
	if len(np.shape(time)) != 1  and type(time) == np.array:
		raise TypeError('The parameter time must be a 1D array.')

	ndat = len(data)
	if dt > 0.0:
		dt_fixed = True
		time = np.linspace(0.,(ndat-1)*dt,ndat)
		fNyq = 1/(2*dt)
	else:
		dt_fixed = False
		# Find the minimum Nyquist frequency and check for fixed time step
		dt0 = time[1]-time[0]
		ddt = 0.0 ; fNyq = 1.E6
		for i in range(ndat-1):
			dt1 = time[i+1]-time[i]
			ddt = max(ddt,abs(dt1-dt0))
			fNyq = min(1/(2*dt1),fNyq)
		 # Check whether time step really is fixed
		if ddt < 1E-8:
			dt_fixed = True
			dt = dt0

	th = TimeHistory(time,data,ordinate=ordinate,fmt=fmt)

	th.setdt(dt,dt_fixed,fNyq)

	return th