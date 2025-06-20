"""
Package: Qtools
Module: energy_spectrum
(C) 2020-2025 Andreas H. Nielsen
See README.md for further details.
"""

import numpy as np
from scipy.integrate import trapz
from math import log10, pi
from qtools.response_spectrum import ResponseSpectrum, _solode
from qtools.config import set_module, Info


@set_module('qtools')
class EnergySpectrum(ResponseSpectrum):
	"""
	General class for energy spectra.

	Parameters
	----------
	x : 1D NumPy array
		A monotonically increasing array of frequency or period values.
	y : 1D NumPy array
		Input energy per unit mass. This must have same length as `x`.
	abscissa : {'f', 'T'}, optional
		Specify quantity contained in x, with 'f' = frequency and 'T' = period.
		Default 'f'.
	xi : float, optional
		Damping ratio. Default 0.05.
	label : str, optional
		Label to use in the legend when plotting the energy spectrum.
		Default '_nolegend_'.
	fmt : str, optional
		The line format to be used when plotting this spectrum (see notes on
		the `fmt` parameter in `Matplotlib documentation
		<https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html>`_.
		With the default value, ``fmt = ''`` (an empty string), the spectrum
		will be plotted with the default matplotlib.pyplot line styles.

	Attributes
	----------
	sa : 1D NumPy array
		Input energy per unit mass in units of J/kg.
	f : 1D NumPy array
		Frequency in units of Hz (1/s).
	fmt : str
		See above under Parameters.
	g : float
		Value of the acceleration due to gravity.
	label : str
		See above under Parameters.
	ndat : integer
		Number of elements in `ei`, `T`, and `f`.
	T : 1D NumPy array
		Period in units of s (:math:`T = 1/f`).
	xi : float
		Damping ratio.

	Notes
	-----
	The following conventions for units are followed:

	* Each instance of EnergySpectrum is defined in terms of standard units.
	* The standard spatial unit is metres.
	* The standard time unit is seconds.
	* The standard mass unit is kilograms.
	* Other units are not supported, but may be used at the discretion of the
	  user.

	When `ei` is an instance of EnergySpectrum, ``str(ei)`` will return a
	string representation of energy spectrum `ei`; this will also work with
	the built-in functions ``format(ei)`` and ``print(ei)``.

	Multiplication is supported: `a*ei` multiplies the spectral ordinates of
	`ei` with `a` where `a` must be an integer or a float.

	"""

	def __init__(self, x, y, abscissa='f', xi=0.05, label='_nolegend_', fmt=''):

		ResponseSpectrum.__init__(self, x, y, abscissa=abscissa, ordinate='ei',
							xi=xi, label=label, fmt=fmt)


@set_module('qtools')
def calcei(th, ffile=None, nf=200, fmin=0.1, fmax=100, xi=0.05):
	"""
	Calculate an energy spectrum from a time history.

	Parameters
	----------
	th : an instance of class TimeHistory
		An acceleration time history defining the base motion.
	ffile : str, optional
		Read frequency points from file `ffile`. If not assigned a value, the
		function will use parameters `nf`, `fmin` and `fmax` to generate an
		array of frequency points. See Notes below.
	nf : int, optional
		Number of frequency points. Default 200.
	fmin : float, optional
		Minimum frequency considered. Default 0.1 Hz.
	fmax : float, optional
		Maximum frequency considered. Default 100.0 Hz.
	xi : float, optional
		Damping ratio. Default 0.05.

	Returns
	-------
	ei : an instance of EnergySpectrum

	Notes
	-----
	If `ffile` is not specified, the function will generate `nf` frequency
	points between `fmin` and `fmax` equally spaced on a logarithmic scale.
	"""

	# Preliminary checks
	if th.ordinate != 'a':
		raise TypeError('The time history used as input to function calcei '
				  'must be an acceleration time history.')
	if xi >= 1.0:
		raise ValueError('The damping ratio must be less than 1.')
	Info.note('Using solver solode to compute energy spectrum.')
	if th.dt_fixed:
		Info.note('The time step is taken as fixed.')
	else:
		Info.note('The time step is taken as variable.')

	if ffile==None:
		# Generate frequency points
		f = np.logspace(log10(fmin),log10(fmax),nf)
	else:
		# Read frequency points from file
		f = np.loadtxt(ffile)
		nf = np.size(f)
		fmax = f[-1]
	if f[0] >= th.fNyq:
		raise ValueError('The lowest frequency of the spectrum is greater than,'
				   ' or equal to, the Nyquist frequency of the time history.')

	w = 2*pi*f[f<=th.fNyq]
	y0 = np.array([0.0, 0.0])

	# Solve
	ei = np.empty_like(w)
	for i in range(w.size):
		sol = _solode(y0, th.time, th.data, xi, w[i], th.dt_fixed)
		# Spectral input energy
		# Calculate the input energy as the sum of kinetic energy, strain
		# energy and energy dissipated in viscuos damping.
		svl = trapz(th.data, x=th.time)
		Ekin = 1/2*(sol[-1,1] + svl)**2
		Estr = 1/2*(w[i]*sol[-1,0])**2
		Evis = trapz(2*xi*w[i]*sol[:,1]**2, x=th.time)
		ei[i] = Evis + Ekin + Estr
		# # Calculate the work done on the system. This will yield similar results.
		# svl = cumtrapz(th.data, th.time, initial=0)
		# intg = -(2*xi*w[i]*sol[:,1] + w[i]**2*sol[:,0])*svl
		# ei[i] = trapz(intg, x=th.time)
		# # Alternatively, calculate the work done on the system by integration
		# # by parts of the total inertia force times ground velocity. Again,
		# # this will yield similar results.
		# svl = cumtrapz(th.data, th.time, initial=0)
		# ei[i] = sol[-1,1]*svl[-1] + svl[-1]**2/2 - trapz(th.data*sol[:,1], x=th.time)

	# Set spectral energies for f > fNyq equal to the value at the fNyq
	if fmax > th.fNyq:
		Info.note('NOTE: Input energies above the Nyquist frequency are '
				'assumed equal to the value at {:6.2f} Hz.'.format(th.fNyq))
		ei = np.append(ei,ei[-1]*np.ones(f.size-ei.size))

	# Complete the creation of an energy spectrum
	Info.end()
	ei = EnergySpectrum(f, ei, abscissa='f', xi=xi)
	ei.setLabel(th.label)
	return ei
