"""
Package: Qtools
Module: random_vibration
(C) 2020-2025 Andreas H. Nielsen
See README.md for further details.
"""

from math import atan2, ceil, exp, isclose, log, pi, sqrt
import numpy as np
from qtools.power_spectrum import PowerSpectrum
from qtools.response_spectrum import ResponseSpectrum
from qtools.config import set_module, Info
from scipy.optimize import brentq
try:
	import pyrvt
except ModuleNotFoundError:
	Info.warn('Package PyRVT not found. Calling function '
			   'pyrvt_rs2ps will have no effect.')
	Info.end()
	PYRVT_PRESENT = False
else:
	PYRVT_PRESENT = True

@set_module('qtools')
class PeakFactorFunction:
	"""
	The peak factor is defined as the ratio :math:`S_A/\sigma_A` where
	:math:`S_A` is the spectral acceleration corresponding to a non-exceedance
	probability :math:`p`, and :math:`\sigma_A` is the standard deviation of
	the response. This class provides functions to evaluate the peak factor.

	Attributes
	----------
	pffun : function
		The peak factor function.
	info : dict
		Information about parameters used to initiate the peak factor
		function. Typical keys are 'smd' (strong motion duration), 'method'
		and 'p' (non-exceedance probability).

	Notes
	-----
	For ordinary use, one of the following class methods should be used to
	create an instance of this class:

	* :meth:`PeakFactorFunction.Vanmarcke_75`
	* :meth:`PeakFactorFunction.Vanmarcke_76`
	* :meth:`PeakFactorFunction.DerKiureghian_79`

	For example, to create a peak factor function according to the Vanmarcke
	(1975) method with strong motion duration of 12 seconds, write::

		>>>pff = qt.PeakFactorFunction.Vanmarcke_75(smd=12)

	Subsequently `pff` can be provided as an argument to
	:func:`qtools.convert2ps` and :func:`qtools.convert2rs`.
	"""

	def __init__(self, pffun, info):

		self.pffun = pffun
		self.info = info

	def __call__(self, Gw, we, wn, xi):
		"""
		Compute the peak factor at the frequencies contained in `wn`.

		Parameters
		----------
		Gw : 1D NumPy array
			Power spectral density values at frequencies contained in `we`.
		we : 1D NumPy array
			Excitation frequencies.
		wn : 1D NumPy array
			Oscillator frequencies.
		xi : float
			Damping ratio.

		Returns
		-------
		rps : ndarray
			Peak factors defined at `wn` frequencies.
		"""

		return self.pffun(Gw, we, wn, xi)

	@staticmethod
	def _moments(G, w, exps):

		lam = [np.trapz(w**n*G, x=w) for n in exps]
		return lam

	@staticmethod
	def _transfer(G, w, wn, xi):
		"""Transfers a power spectrum.
		Input: total base acceleration --> output: relative displacement."""

		Hsq = 1/((wn**2 - w**2)**2 + (2*xi*wn*w)**2)
		return G*Hsq

	@staticmethod
	def _enrich(G, we, wn, xi, return_index=False, kind='log'):
		"""
		Creates data points that are more closely spaced than the points given
		in `(we, G)` over an interval equal to `5*xi*wn` centred on `wn`.
		Additional points are created only if the step size in `we` is greater
		than `xi*wn/3` (the bandwidth divided by 6). This method also enforces
		`Gr` >= 0.

		Parameters
		----------
		G : ndarray
			The y-coordinates of the data points (spectral densities), same
			length as `we`.
		we : ndarray
			The x-coordinates of the data points (frequencies). It is assumed
			that frequencies in `we` are equi-spaced.
		wn : float
			Centre frequency.
		xi : float
			Damping ratio.
		return_index : bool, optional
			If True, also return the index `i` for which ``wer[i] == wn`` in
			the enriched data.
		kind : {'lin', 'log'}, optional
			Generate evenly spaced numbers on a linear ('log') or logarithmic
			('log') scale. Default 'log'.

		Returns
		-------
		Gr : ndarray
			Enriched y-coordinates corresponding to `wer`.
		wer : ndarray
			Enriched x-coordiates with additional points centered around `wn`.
		index : int
			Only provided if `return_index` is True.
			Is equal to the index `i` for which ``wer[i] == wn`` if
			enrichment has been performed.
			Is equal to -1 if no enrichment has been performed. This happens
			when the step size in `we` is less than `xi*wn/3`, in which case
			`Gr` = `G` and `wer` = `we` are returned.
		"""

		dw = we[1]-we[0]
		bw = 2*xi*wn
		m = 6
		index = -1
		if dw <= bw/m:
			Gr = G
			wer = we
		else:
			wl = we[we < wn - 5/2*bw]
			wu = we[we > wn + 5/2*bw]
			if kind=='lin':
				# Linspace
				N = ceil(m*(wu[0] - wn)/bw)
				wlu = np.linspace(wn, wu[0], num=N, endpoint=False)
			else:
				# Logspace
				N = ceil(log(wu[0]/wn)/log(1 + bw/(m*wn)))
				wlu = np.geomspace(wn, wu[0], num=N, endpoint=False)
			wll = np.flip(2*wn - wlu)[:-1]
			wer = np.concatenate((wl,wll,wlu,wu))
			index = wl.size + wll.size

			# Interpolate at new enriched frequencies
			Gr = np.interp(wer, we, G)

		# Enforce G >= 0:
		Gr = np.where(Gr < 0, 0, Gr)

		if return_index:
			return (Gr, wer, index)
		else:
			return (Gr, wer)

	@staticmethod
	def FR(s, vt, de, ic=0):
		r"""
		Returns the cumulative distribution of the maximum absolute value of
		a stationary Gaussian process with zero mean according to
		`Vanmarcke (1975)`_.

		Parameters
		----------
		s : float or 1D NumPy array
			The normalised response barrier.
		vt : float
			The product of the mean crossing rate and the duration
			(see notes below).
		de : float
			The effective spread / dispersion of the process (see notes below).
		ic : int, optional
			Initial condition: `ic` = 0 signifies that the process starts
			from zero so that the probability of starting below the response
			barrier is 1; any other `ic` value signifies a random initial
			condition.

		Returns
		-------
		Float or 1D NumPy array
			The value(s) of the cumulative distribution of the maximum
			absolute value of a stationary Gaussian process.

		Notes
		-----
		This function implements eqn (29) in `Vanmarcke (1975)`_.

		The normalised response level `s` is defined as:

		.. math::

			s = \frac{r}{\lambda_0}

		where :math:`r` is the peak response and :math:`\lambda_0` is the 0th
		moment of the spectral density function of the response (i.e. the root
		mean square).

		The parameter `vt` should be evaluated as the product of the mean
		zero crossing rate :math:`\nu` and the strong-motion duration
		:math:`T_D`. The mean zero crossing rate can be evaluated as:

		.. math::

			\nu = \frac{1}{\pi}\sqrt{\frac{\lambda_2}{\lambda_0}}

		where :math:`\lambda_2` is the 2nd moment of the spectral density
		function. Note: :math:`\nu = 2\nu_0` where :math:`\nu_0` is the mean
		zero up-crossing rate.

		The parameter `de` should be evaluated as:

		.. math::

			\delta_e = \left(\sqrt{1-\frac{\lambda_1^2}{\lambda_0 \lambda_2}}\right)^{1.2}

		"""

		# Note: when `s` is a float, the following implementation is faster
		# than the unified NumPy implementation.
		if isinstance(s, float):
			if isclose(s, 0):
				f = 0
			else:
				if ic==0:
					A = 1
				else:
					A = 1 - exp(-s**2/2)
				f = A*exp(-vt*(1 - exp(-sqrt(pi/2)*de*s))/(exp(s**2/2) - 1))

		elif isinstance(s, np.ndarray):
			if ic==0:
				A = np.ones_like(s)
			else:
				A = (1 - np.exp(-s**2/2))
			f = np.zeros_like(s)
			nz = np.nonzero(s)
			f[nz] = A[nz]*np.exp(-vt*(1 - np.exp(-sqrt(pi/2)*de*s[nz]))
                                     / (np.exp(s[nz]**2/2) - 1))
		else:
			raise TypeError('Unsupported type of s: {}'.format(type(s)))

		return f

	@classmethod
	def Vanmarcke_75(cls, smd=10, p=0.5, ic=0):
		"""
		Creates a peak factor function based on the the cumulative distribution
		of the maximum absolute value of a stationary Gaussian process as
		per `Vanmarcke (1975)`_.

		Parameters
		----------
		smd : float
			Strong motion duration. Default 10 seconds.
		p : float
			Non-exceedance probability (0 < p < 1). Default 0.5 corresponding
			to the median value. Alternatively, the `p` parameter can be set
			equal to 'mean', in which case the function returns the mean value.
		ic : int, optional
			Initial condition: `ic` = 0 signifies that the process starts
			from zero so that the probability of starting below the response
			barrier is 1; any other `ic` value signifies a random initial
			condition.

		Returns
		-------
		A instance of :class:`PeakFactorFunction`.

		References
		----------
		.. _Vanmarcke (1975):

		Vanmarcke, E.M, 1975: "On the Distribution of the First-Passage Time
		for Normal Stationary Random Process", *Journal of Applied Mechanics*,
		Vol. 42, pp. 215-220.
		"""

		def pffun(Gw, we, wn, xi):

			rps = np.empty_like(wn)
			xis = xi/(1 - np.exp(-2*xi*wn*smd))
			if p=='mean':
				s = np.linspace(0, 6, num=100)

			for i in range(wn.size):
				Gr, wer = cls._enrich(Gw, we, wn[i], xis[i])
				Gyr = cls._transfer(Gr, wer, wn[i], xis[i])
				ly0, ly1, ly2 = cls._moments(Gyr, wer, (0, 1, 2))
				nu = sqrt(ly2/ly0)/pi
				de = (sqrt(1 - ly1**2/(ly0*ly2)))**1.2
				if p=='mean':
					rps[i] = np.trapz(1 - cls.FR(s, nu*smd, de, ic=ic), dx=s[1])
				else:
					rps[i] = brentq(lambda x : cls.FR(x, nu*smd, de, ic=ic) - p, 0, 6)

			return rps

		info = {'smd': smd, 'method': 'Vanmarcke_75', 'p': p, 'ic': ic}

		return cls(pffun, info)

	@classmethod
	def Vanmarcke_76(cls, smd=10, p=0.5, opt=0):
		"""
		Creates a peak factor function according to the method outlined by
		`Vanmarcke (1976)`_.

		Parameters
		----------
		smd : float
			Strong motion duration. Default 10 seconds.
		p : float
			Non-exceedance probability (0 < p < 1). Default 0.5 corresponding
			to the median value.

		Returns
		-------
		A instance of :class:`PeakFactorFunction`.

		Notes
		-----
		The :meth:`Vanmarcke_75` method should be used in preference to this
		method. The two methods yield nearly identical results except at low
		frequencies and/or low probabilities where the approximations of
		:meth:`Vanmarcke_76` appear to lose validity.

		References
		----------
		.. _Vanmarcke (1976):

		Vanmarcke, E.M, 1976: "Structural Response to Earthquakes", Chapter 8
		in: *Seismic Risk and Engineering Decisions*, eds. C. Lomnitz & E.
		Rosenblueth, Elsevier.
		"""

        # TO-DO: Decide if option 0 or 1 should be only option.
        # Option 0 was the original default option
        # Option 1 was implemented as a new development for testing, but
        # should not be used at this stage.

		if opt==0:
			def pffun(Gw, we, wn, xi):

				rps = np.ones_like(wn)
				m = (1 - np.exp(-2*xi*wn*smd))/(1 - np.exp(-xi*wn*smd))
				smd0 = smd*np.exp(-2*(m - 1))
				xis = xi/(1 - np.exp(-2*xi*wn*smd))

				for i in range(wn.size):
					Gr, wer = cls._enrich(Gw, we, wn[i], xis[i])
					Gyr = cls._transfer(Gr, wer, wn[i], xis[i])
					ly0, ly1, ly2 = cls._moments(Gyr, wer, (0, 1, 2))
					Omgy = sqrt(ly2/ly0)
					# Need check on math domain here.
					de = (sqrt(1 - ly1**2/(ly0*ly2)))**1.2
					n = max((Omgy*smd0[i]/(2*pi))/(-log(p)), 0.5)
					arg = 2*n*(1 - exp(-de*sqrt(pi*log(2*n))))
					if arg < exp(0.5):
						Info.deb('arg < exp(0.5) = 1.65 in Vanmarcke_76 peak '
						  'factor function at f = {} Hz.'.format(wn[i]/(2*pi)))
						# The following is completely heuristic and might be
						# improved:
						arg = (max(2*n, exp(0.5)) - exp(0.5))*de + exp(0.5)
						Info.note('Assuming arg = {:.2f}. Check results '
									   'carefully.'.format(arg))
					rps[i] = sqrt(2*log(arg))

				return rps

		elif opt==1:
			def pffun(Gw, we, wn, xi):

				rps = np.ones_like(wn)
				l0, l1, l2 = cls._moments(Gw, we, (0, 1, 2))
				Omg = sqrt(l2/l0)
				d = (sqrt(1 - l1**2/(l0*l2)))**1.2

				for i in range(wn.size):
					xis = xi/(1 - np.exp(-2*xi*wn[i]*smd))
					Gr, wer = cls._enrich(Gw, we, wn[i], xis)
					Gyr = cls._transfer(Gr, wer, wn[i], xis)
					ly0, ly1, ly2 = cls._moments(Gyr, wer, (0, 1, 2))

					xis2 = xi/(1 - np.exp(-xi*wn[i]*smd))
					Gr, wer = cls._enrich(Gw, we, wn[i], xis2)
					Gyr = cls._transfer(Gr, wer, wn[i], xis2)
					ly0s2 = cls._moments(Gyr, wer, (0,))[0]

					m = ly0/ly0s2
					smd0 = smd*exp(-2*(m-1))

					Omgy = sqrt(ly2/ly0)
					de = (sqrt(1 - ly1**2/(ly0*ly2)))**1.2
					Info.deb('Tn = {}, Omg = {}, {}, {}'.format(2*pi/wn[i], Omg, Omgy, wn[i]))
					Info.deb('m = {}, de = {}, {}, {}'.format(m, d, de, sqrt(4*xis/pi)**1.2))
					n = max((Omgy*smd0/(2*pi))/(-log(p)), 0.5)
					arg = 2*n*(1 - exp(-de*sqrt(pi*log(2*n))))
					if arg < exp(0.5):
						Info.deb('arg < exp(0.5) = 1.65 in Vanmarcke_76 peak '
						  'factor function at f = {} Hz.'.format(wn[i]/(2*pi)))
						# The following is completely heuristic and might be
						# improved:
						arg = (max(2*n, exp(0.5)) - exp(0.5))*de + exp(0.5)
						Info.note('Assuming arg = {:.2f}. Check results '
									   'carefully.'.format(arg))
					rps[i] = sqrt(2*log(arg))

				return rps

		info = {'smd': smd, 'method': 'Vanmarcke_76', 'p': p}

		return cls(pffun, info)

	@classmethod
	def DerKiureghian_79(cls, smd=10):
		"""
		Creates a peak factor function according to the approximation derived
		by `Der Kiureghian (1979)`_. Only the expected (mean) value is
		calculated by this function.

		Parameters
		----------
		smd : float
			Strong motion duration. Default 10 seconds.

		Returns
		-------
		A instance of :class:`PeakFactorFunction`.

		References
		----------
		.. _Der Kiureghian (1979):

		Der Kiureghian, A, 1979: *On Response of Structures to Stationary
		Excitation*, Report No. UBC/EERC-79/32, Earthquake Engineering
		Research Centre, University of California, Berkeley.
		"""

		def pffun(Gw, we, wn, xi):

			rps = np.ones_like(wn)
			xis = xi/(1 - np.exp(-2*xi*wn*smd))

			for i in range(wn.size):
				Gr, wer = cls._enrich(Gw, we, wn[i], xis[i])
				Gyr = cls._transfer(Gr, wer, wn[i], xis[i])
				ly0, ly1, ly2 = cls._moments(Gyr, wer, (0, 1, 2))
				Omgy = sqrt(ly2/ly0)
				delt = sqrt(1-ly1**2/(ly0*ly2))
				nu = Omgy/pi
				if delt < 0.69:
					nue = (1.63*delt**0.45 - 0.38)*nu
				else:
					nue = nu
				dkp = sqrt(2*log(nue*smd)) + 0.5772/sqrt(2*log(nue*smd))
				rps[i] = dkp

			return rps

		info = {'smd': smd, 'method': 'DerKiureghian_79', 'p': 'mean'}

		return cls(pffun, info)

# Define an internal interpolator function
def _varinterp(w, wn, Var):
	# This function assumes w >= 0 and wn >= 0 and w[0] != wn[0].
	if np.any(w < wn[0]): # == np.all(wn > 0)
		V0 = Var[0]*w[w < wn[0]]**2/wn[0]**2
		V1 = np.exp(np.interp(np.log(w[w >= wn[0]]), np.log(wn), np.log(Var)))
		V = np.concatenate((V0,V1))
	else: # np.all(w >= wn[0]) == np.all(w > 0)
		V = np.exp(np.interp(np.log(w), np.log(wn[wn > 0]),
						np.log(Var[wn > 0])))

	return V

@set_module('qtools')
def convert2ps(rs, pff, nf=2049, method=0, esnv=True):
	"""
	Converts a response spectrum to a compatible power spectrum.

	Parameters
	----------
	rs : an instance of :class:`ResponseSpectrum`
		The response spectrum that is to be converted.
	pff : an instance of :class:`PeakFactorFunction`
		The peak factor function used in the conversion process.
	nf : int, optional
		Number of frequency points used for the power spectrum computation.
		This should be greater than `rs.ndat`.
	method : int, optional
		Method used for spectral conversions. See :func:`.calc_comp_ps`
		for further details. Default 0.
	esnv : bool, optional
		Eliminate spurious negative values. Default True.

	Returns
	-------
	ps : an instance of :class:`PowerSpectrum`
	"""

	lbl = rs.label+' ' if rs.label != '_nolegend_' else ''
	Info.note('Converting response spectrum {}to a power spectrum using '
		   'method {}.'.format(lbl,method))
	if method >= 1:
		Info.note('Method {} can take some time to complete...'.format(method))

	# Get the target accelerations, oscillator frequencies and damping value
	sa = rs.sa*rs.g
	fn = rs.f
	wn = 2*pi*fn
	xi = rs.xi
	smd = pff.info['smd']

	# Initial values for the peak factor
	rps = 2.5*np.ones_like(sa)

	# Initial power spectrum with excitation frequencies `we`
	we = np.linspace(0, 4*pi*fn[-1], num=nf)
	G0 = np.ones_like(we)

	# Effective damping values
	# Note: xis[i] corresponds to we[i+1]; xis[0] is not needed.
	xis = xi/(1-np.exp(-2*xi*we[1:]*smd))

	# Set the parameters for the iterative procedure
	Nit = 0
	maxNit = 10
	DG0 = 1E6
	nonConvCount = 0
	converged = False

	# Calculate the power spectrum by iteration
	while not converged and Nit < maxNit:
		# Iteration number
		Nit += 1
		# Calculate the pseudo-acceleration variance
		Var = (sa/rps)**2
		# Interpolate the variance at the excitation frequencies
		Vari = _varinterp(we, wn, Var)
		# Calculate the compatible power spectrum
		G1 = calc_comp_ps(Vari, we, xis, method=method)
		# Calculate the maximum absolute relative change
		DG1 = np.amax(np.abs(G1-G0))/np.amax(G1)
		# Convergence checks
		conv = DG1 - DG0
		if conv > 0: nonConvCount += 1
		if nonConvCount > 0:
			Info.warn('The convert2ps algorithm appears to be '
				 'diverging in {} iteration(s).'.format(nonConvCount))
		if DG1 < 1E-3: converged = True
		# Prepare for next iteration
		G0 = G1
		DG0 = DG1
		if not converged:
			# Get the new peak factor spectrum
			rps = pff(G1, we, wn, xi)

	if not converged:
		Info.warn('In convert2ps: Reached the maximum number of '
				'iterations without convergence.')
	else:
		Info.note('The conversion algorithm converged in {} '
			'iterations.'.format(Nit))

	if esnv and np.any(G1 < 0):
		G1 = np.where(G1 < 0, 0, G1)
		Info.note('Eliminated some spurious negative values in the computed '
			'power spectrum')

	Info.end()
	return PowerSpectrum(we/(2*pi), 2*pi*G1, unit='m**2/s**3', label=rs.label,
					  fmt=rs.fmt)

@set_module('qtools')
def convert2rs(ps, pff, xi=0.05, nf=200, fmin=0.1, fmax=100, method=0):
	"""
	Converts a power spectrum to a compatible response spectrum.

	Parameters
	----------
	ps : an instance of :class:`PowerSpectrum`
		The power spectrum that is to be converted.
	pff : an instance of :class:`PeakFactorFunction`
		The peak factor function used in the conversion process.
	xi : float, optional
		The damping ratio of the response spectrum.
	nf : int, optional
		Number of frequency points. Default 200. The response spectrum will be
		defined at `nf` frequency points between `fmin` and `fmax` equally
		spaced on a logarithmic scale.
	fmin : float, optional
		Minimum frequency considered. Default 0.1 Hz.
	fmax : float, optional
		Maximum frequency considered. Default 100.0 Hz.
	method : int
		Method used for spectral conversions. See :func:`.calc_comp_Var`
		for further details.

	Returns
	-------
	rs : an instance of :class:`ResponseSpectrum`
	"""

	lbl = ps.label+' ' if ps.label != '_nolegend_' else ''
	Info.note('Converting power spectrum {}to a response spectrum.'.format(lbl))

	if not isclose(ps.w[0], 0.0):
		Info.warn('The lowest frequency of the power spectrum should be close '
			'or equal to zero. The computed response spectrum may be invalid.')

	# Strong motion duration
	smd = pff.info['smd']

	# Effective damping values
	# Note: xis[i] corresponds to we[i+1]; xis[0] is not needed.
	xis = xi/(1-np.exp(-2*xi*ps.w[1:]*smd))

	# Initialise response spectrum frequencies
	fn = np.geomspace(fmin, fmax, nf)
	wn = 2*pi*fn

	# Calculate the compatible variance spectrum
	Var = calc_comp_Var(ps.Gw, ps.w, xis, method=method)

	# Interpolate the variance at the response spectrum frequencies
	Vari = _varinterp(wn, ps.w, Var)

	# Calculate the peak factors
	rps = pff(ps.Gw, ps.w, wn, xi)

	# Compute the spectral accelerations
	sa = rps*np.sqrt(Vari)

	Info.end()
	return ResponseSpectrum(fn, sa, ordinate='sa', xi=xi, label=ps.label)

def calc_comp_ps(Var, wn, xi, method=0, Gin=None, maxIter=100):
	"""
	Calculates a power spectrum that is compatible with a given variance of
	response.

	Parameters
	----------
	Var : 1D NumPy array
		The variance of the oscillator response.
	wn : 1D NumPy array
		Oscillator frequencies (rad/s). It is assumed that `wn` is equi-spaced
		with ``wn[0] == 0``. This array is also used to define the excitation
		frequencies for the output array `G`.
	xi : 1D NumPy array or float
		Damping ratio. For time-limited excitation, this should be the
		fictitious time-dependent damping ratio (in an array whose size is
		equal to ``wn.size - 1``). For steady-state excitation,
		this is the damping ratio of the oscillator.
	method : int
		Method used to compute the power spectrum. The following are
		supported:

			0. The method proposed by `Vanmarcke (1976)`_. This is relatively
			   fast and accurate.
			1. An iterative method that is slower than method 0. Corresponds
			   to method 1 in :func:`.calc_comp_Var`.
			2. An iterative method that is slower than methods 0 and 1.
			   Corresponds to method 2 in :func:`.calc_comp_Var`. In theory,
			   this is the most accurate method, but it can result in high-
			   frequency oscillations in the resulting power spectrum,
			   either because the method is numerically ill-conditioned or
			   because the exact solution (if it exists) is not smooth.

	Gin : 1D NumPy array
		Initial estimate of G. Used only for methods 1 and 2. If not supplied,
		method 0 will be used to compute an initial estimate.
	maxIter : int
		Maximum number of iterations used in methods 1 and 2. Not applicable
		in method 0.

	Returns
	-------
	G : 1D NumPy array
		The compatible spectral density function (power spectrum).
	"""

	if isinstance(xi, float):
		xis = xi*np.ones(wn.size-1)
	else:
		xis = xi
	G = np.zeros_like(wn)
	dw = wn[1]

	enrich = PeakFactorFunction._enrich
	transfer = PeakFactorFunction._transfer

	def met0():
		for i in range(wn.size-1):
			A = wn[i+1]*(pi/(4*xis[i]) - 1) + dw/2
			G[i+1] = (Var[i+1] - (G[i]*dw/2 + np.trapz(G[:i+1], dx=dw)))/A

	def met1():
		for i in range(wn.size-1):
			# Note: index i+1 corresponds to index j in the enriched
			# arrays so that wr[j] = wn[i+1], unless j = -1 (indicating
			# no enrichment), in which case we simply set j = i+1.
			Gr, wr, j = enrich(G, wn, wn[i+1], xis[i],
				return_index=True)
			if j==-1:
				j = i+1
			Fr = wn[i+1]**4*transfer(Gr, wr, wn[i+1], xis[i])
			sq = sqrt(1 - xis[i]**2)
			bj = wn[i+1]/8*(pi/xis[i] - log((1 + sq)/(1 - sq))/sq)
			dw = wr[j] - wr[j-1]
			A = bj + dw/(8*xis[i]**2)
			G[i+1] = (Var[i+1] - (Fr[j-1]*dw/2 + np.trapz(Fr[:j], x=wr[:j])))/A

	def met2(G0):
		G1 = G0.copy()
		alpha = 0.25
		wm = wn[-1]
		for i in range(wn.size-2):
			# Note: see above under met1().
			x = xis[i]
			sq = sqrt(1-x**2)
			Gr, wr, j = enrich(G1, wn, wn[i+1], x, return_index=True)
			if j==-1:
				j = i+1
			Fr = wn[i+1]**4*transfer(Gr, wr, wn[i+1], x)
			dw0 = wr[j] - wr[j-1]
			dw1 = wr[j+1] - wr[j]
			t1 = (pi/(4*x) - 1/(4*x)*atan2(2*wm/wn[i+1]*x, 1-(wm/wn[i+1])**2))
			t2 = log((wm*(wm + 2*sq*wn[i+1]) + wn[i+1]**2)/
						 (wm*(wm - 2*sq*wn[i+1]) + wn[i+1]**2))/(8*sq)
			R = G0[-1]*wn[i+1]*(t1 - t2)
			Intl = np.trapz(Fr[:j], x=wr[:j])
			Intu = np.trapz(Fr[j+1:], x=wr[j+1:])
			A = (dw0 + dw1)/(8*xis[i]**2)
			G1[i+1] = ((1-alpha)*G0[i+1] + alpha*(Var[i+1]
					- (Intl + Fr[j-1]*dw0/2 + Fr[j+1]*dw1/2 + Intu + R))/A)
			if G1[i+1] < 0: G1[i+1] = 0
		# Last step
		x = xis[-1]
		sq = sqrt(1-x**2)
		Fr1 = wn[-1]**4*transfer(G1, wn, wn[-1], x)
		dw = wn[-1] - wn[-2]
		bj = wn[-1]/8*(pi/x - log((1+sq)/(1-sq))/sq)
		G1[-1] = (Var[-1] - (np.trapz(Fr1[:-1], x=wn[:-1])
				+ Fr1[-2]*dw/2))/(bj + dw/(4*xis[-1]**2))
		return G1

	if method==0:
		met0()

	else:
		nonConverged = True
		itno = 0
		while nonConverged and itno < maxIter:
			G0 = G.copy()
			if itno==0 and Gin is None:
				# Use method 0 to create an initial estimate of G
				met0()
			else:
				if method==1:
					met1()
				else:
					G = met2(G)
			itno +=1
			maxreldiff = np.amax(np.fabs(G-G0)/np.amax(G))
			if maxreldiff < 1E-4: nonConverged = False
			Info.deb('Iteration: {}, max rel diff: {}'.format(itno,maxreldiff))

	return G

def calc_comp_Var(G, we, xi, method=2):
	"""
	Calculates a variance spectrum that is compatible with a given power
	spectrum.

	Parameters
	----------
	G : 1D NumPy array
		The input power spectrum. It is assumed that ``G[0] == 0``.
	we : 1D NumPy array
		Excitation frequencies (rad/s). It is assumed that `we` is equi-spaced
		with ``we[0] == 0``. This array is also used to define the oscillator
		frequencies for the output array `Var`.
	xi : 1D NumPy array or float
		Damping ratio. For time-limited excitation, this should be the
		fictitious time-dependent damping ratio (in an array whose size is
		equal to ``wn.size - 1``). For steady-state excitation,
		this is the damping ratio of the oscillator.
	method : int
		Method used to compute the variance spectrum. The following are
		supported:

			0. The method proposed by `Vanmarcke (1976)`_. This is the exact
			   inverse of method 0 in :func:`.calc_comp_ps`.
			1. An improved method corresponding to method 1 in
			   :func:`.calc_comp_ps`.
			2. An improved method corresponding to method 2 in
			   :func:`.calc_comp_ps`. This the most accurate method and
			   therefore the default.
			3. A method similar to method 2, but without enrichment of data
			   points. In theory less accurate at low and high frequencies,
			   but probably the fastest. There is no corresponding method in
			   :func:`.calc_comp_ps` (since the inverse solution is
			   ill-conditioned).

	Returns
	-------
	Var : 1D NumPy array
		The compatible variance spectrum.
	"""

	if isinstance(xi, float):
		xis = xi*np.ones(we.size-1)
	else:
		xis = xi
	Var = np.zeros_like(we)
	dw = we[1]

	if method==0:
		for i in range(we.size-1):
			Var[i+1] = (G[i+1]*we[i+1]*(pi/(4*xis[i])-1)
					    + np.trapz(G[0:i+2], dx=dw))
	elif method==1 or method==2:
		enrich = PeakFactorFunction._enrich
		transfer = PeakFactorFunction._transfer
		wm = we[-1]
		for i in range(we.size-1):
			# Note: index i+1 corresponds to index j in the enriched arrays
			# so that wr[j] = wn[i+1], unless j == -1 (indicating no
			# enrichment), in which case we simply set j = i+1.
			wn = we[i+1]
			x = xis[i]
			sq = sqrt(1-x**2)
			if method==1:
				Gr, wr, j = enrich(G, we, wn, x, return_index=True)
				if j==-1:
					j = i+1
				Fr = wn**4*transfer(Gr, wr, wn, x)
				bj = wn/8*(pi/x - log((1 + sq)/(1 - sq))/sq)
				Var[i+1] = np.trapz(Fr[:j+1], x=wr[:j+1]) + Gr[j]*bj
			else:
				Gr, wr = enrich(G, we, wn, x)
				Fr = wn**4*transfer(Gr, wr, wn, x)
				t1 = (pi/(4*x) - 1/(4*x)*atan2(2*wm/wn*x, 1-(wm/wn)**2))
				t2 = log((wm*(wm + 2*sq*wn) + wn**2)/
						 (wm*(wm - 2*sq*wn) + wn**2))/(8*sq)
				Var[i+1] = np.trapz(Fr, x=wr) + Gr[-1]*wn*(t1 - t2)
	elif method==3:
		N = xis.size
		assert N == we.size-1, 'Unexpected size of input arrays'
		we = np.reshape(we[1:], (1,N))
		xis = np.reshape(xis, (N,1))
		wn = we.T
		A = dw*wn**4/((wn**2 - we**2)**2 +(2*xis*wn*we)**2)
		G[-1] /= 2
		Var[1:] = A@G[1:]
	else:
		raise ValueError('Method {} is not supported.'.format(method))

	return Var


@set_module('qtools')
def pyrvt_rs2ps(rs, method='V75', Td=None, Mw=6, R=30, region='cena'):
	"""Converts a response spectrum into a power spectrum using Albert R.
	Kottke's pyRVT package.

	Parameters
	----------
	rs : an instance of class :class:`.ResponseSpectrum`
		The response spectrum that is to be converted.
	method : str, optional
		Method (see notes below).
	Td : float, optional
		Duration (see notes below). If no duration is provided, then pyRVT will
		estimate a value.
	Mw : float, optional
		Earthquake magnitude (see notes below).
	R : float, optional
		Earthquake distance in km (see notes below).
	region : {'wna', 'cena'}, optional
		Region (see notes below).

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
	refer to the documentation for `PyRVT <https://pyrvt.readthedocs.io/>`_.

	If pyRVT is not installed on the system, the function will do
	nothing and return None. To use the function, the main module must be
	protected by an ``if __name__ == '__main__':`` statement.

	Version 0.7.2 of pyRVT supports the follow peak factor methods:

		* BJ84: Boore and Joyner (1984)
		* BT12: Boore and Thompson (2012)
		* BT15: Boore & Thompson (2015)
		* CLH56: Cartwright & Longuet-Higgins (1956)
		* D64: Davenport (1964)
		* DK85: Der Kiureghian (1985)
		* LP99: Liu and Pezeshk (1999)
		* TM87: Toro and McGuire (1987)
		* WR18: Wang & Rathje (2018)
		* V75: Vanmarcke (1975)

	The parameters `Mw`, `R` and `region` are only used by BT12, BT15,
	and WR18.
	"""

	if PYRVT_PRESENT:
		Info.note('Remember to protect the main module with'
				' an if __name__ == \'__main__\': statement')

		if Td is None:
			TdNotSpecified = True
		else:
			TdNotSpecified = False

		e = {}
		e['psa'] = rs.sa
		e['magnitude'] = Mw
		e['distance'] = R
		e['region'] = region
		e['duration'] = Td
		freqs = pyrvt.tools.calc_compatible_spectra(method, rs.T, [e], damping=rs.xi)

		Td = e['duration']
		if TdNotSpecified:
			Info.note('The duration determined by pyRVT is {:5.2f} sec'.format(Td))

		X = e['fa']*9.81
		if np.any(np.iscomplex(X)):
			Info.debug('Found complex numbers in X')

		Wf = 2*X**2/Td
		ps = PowerSpectrum(freqs, Wf, unit='m**2/s**3')
		rsc = ResponseSpectrum(rs.f, e['psa_calc'], abscissa='f')

		return (ps, rsc)





