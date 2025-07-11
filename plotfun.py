"""
Package: Qtools
Module: plotfun
(C) 2020-2025 Andreas H. Nielsen
See README.md for further details.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from qtools.config import set_module, Info
from qtools.power_spectrum import PowerSpectrum, FourierSpectrum
from math import log10

# Define a formatter for tick labels
def format_func(x, pos):
	"""
	Formats tick labels like Excel's general format.
	"""
	if x < 1:
		pr = int(-log10(x))
		return '{val:.{prec}f}'.format(val=x, prec=pr)
	else:
		return '{val:.0f}'.format(val=x)

@set_module('qtools')
def plotrs(*args, **kwargs):
	"""Function for plotting instances of class ResponseSpectrum.

	Parameters
	----------
	*args
		Any number of response spectra
	**kwargs
		Optional named parameters (see below under **Other parameters**)

	Other parameters
	----------------
	style : str
		Plot style. See `Matplotlib style sheets reference <https://
		matplotlib.org/stable/gallery/style_sheets/
		style_sheets_reference.html>`_ for available styles. Default `default`.
	xaxis : {'f', 'T'}
		Quantity to plot on the x-axis. Default 'f'.
	yaxis : {'sa', 'sv', 'sd'}
		Quantity to plot on the y-axis. Default 'sa'.
	xscale : {'log', 'lin'}
		Specifies the scale on the x-axis (logarithmic or linear).
		Default 'log'.
	legend : dict (or bool)
		The keys in this dictionary will be used to set the arguments in a
		call to :func:`matplotlib.pyplot.legend`. If this parameter is not
		provided, the legend will not be shown. This parameter can also be set
		to True to show the legend with default parameters ``{'loc': 'best'}``.
	filename : str
		If given, save plot to file using `filename` as file name. The file
		name should include the desired extension (e.g. 'png' or 'svg'), from
		which the file format will be determined as per
		:func:`matplotlib.pyplot.savefig`.
	transparent : bool
		This parameter will be passed directly to
		:func:`matplotlib.pyplot.savefig` (if a `filename` is given). Default
		``False``.
	dpi : int
		Dots per inch to use if saving plot to file. Default None.
	right : float
		Sets the upper limit on the x-axis.
	left : float
		Sets the lower limit on the x-axis.
	top : float
		Sets the upper limit on the y-axis.
	bottom : float
		Sets the lower limit on the y-axis.
	grid : dict
		The keys in this dictionary will be used to set the arguments in a
		call to :func:`matplotlib.pyplot.grid`. Default: ``{'which': 'major',
		'color': '0.75'}``
	fig_size : (float, float)
		Set the figure size. The first value in the tuple is the figure width
		(in cm) and the second is the figure height (also in cm). If None is
		passed, the default Matplotlib values will be used. Default: None.
	title : str
		Title to be placed above plot.

	Notes
	-----
	The line format can be set in the 'fmt' attribute of a response spectrum.
	The line format is passed directly to :func:`matplotlib.pyplot.plot` as
	'fmt'. Additionally, the color can be set in
	:meth:`qtools.ResponseSpectrum.setLineFormat`.

	Examples
	--------
	A typical call with 3 response spectra and default options is::

		qt.plotrs(rs1,rs2,rs3)

	If the spectra are contained in a list (say, `rslist`), unpack the list
	inside the call using the star operator::

		qt.plotrs(*rslist)

	"""
	# Get the parameters
	style = kwargs.get('style', 'default')
	xaxis = kwargs.get('xaxis', 'f')
	yaxis = kwargs.get('yaxis', 'sa')
	xscale = kwargs.get('xscale', 'log')
	yscale = kwargs.get('yscale', 'lin')
	if 'legend' in kwargs:
		show_legend = True
		legend = {'loc': 'best'}
		if isinstance(kwargs['legend'], dict):
			legend = kwargs['legend']
	else:
		show_legend = False
	filename = kwargs.get('filename', '')
	transparent = kwargs.get('transparent', False)
	dpi = kwargs.get('dpi', None)
	grid = kwargs.get('grid', {'which': 'major', 'color': '0.75'})
	fig_size = kwargs.get('fig_size', None)
	title = kwargs.get('title', None)

	plt.style.use(style)

	# Get the axes of a new plot
	fig, ax = plt.subplots()

	# Set the appropriate type of plot
	if xscale == 'log' and yscale == 'log':
		plot = ax.loglog
	elif xscale == 'log' and yscale == 'lin':
		plot = ax.semilogx
	elif xscale == 'lin' and yscale == 'log':
		plot = ax.semilogy
	elif xscale == 'lin' and yscale == 'lin':
		plot = ax.plot

	# Set the labels on the axes
	if xaxis == 'f':
		ax.set_xlabel('Frequency [Hz]')
	elif xaxis == 'T':
		ax.set_xlabel('Period [s]')
	else:
		raise ValueError('{} is not a valid value for the x-axis on a'
				   ' response spectrum plot'.format(xaxis))
	if yaxis == 'sa':
		ax.set_ylabel('Spectral acceleration [g]')
		ucf = 1.0
	elif yaxis == 'sv':
		ax.set_ylabel('Spectral velocity [cm/s]')
		ucf = 100
	elif yaxis == 'sd':
		ax.set_ylabel('Spectral displacement [mm]')
		ucf = 1000
	elif yaxis == 'ei':
		ax.set_ylabel('Spectral input energy [J/kg]')
	else:
		raise ValueError('{} is not a valid value for the y-axis on a'
				   ' response spectrum plot'.format(yaxis))

	xmax = -1e6
	xmin = 1e6
	for rs in args:
		x = rs.__dict__[xaxis]
		y = rs.__dict__[yaxis]
		xmax = max(np.amax(x),xmax)
		xmin = min(np.amin(x),xmin)
		if rs.color == '':
			plot(x, ucf*y, rs.fmt, label=rs.label)
		else:
			plot(x, ucf*y, rs.fmt, label=rs.label, color=rs.color)

	# Set upper limit on x-axis if specified
	if 'right' in kwargs:
		xmax = kwargs['right']
		ax.set_xlim(right=xmax)

	# Set lower limit on x-axis if specified
	if 'left' in kwargs:
		xmin = kwargs['left']
		ax.set_xlim(left=xmin)

	# Set upper limit on y-axis if specified
	if 'top' in kwargs:
		ax.set_ylim(top=kwargs['top'])

	# Set lower limit on y-axis if specified
	if 'bottom' in kwargs:
		ax.set_ylim(bottom=kwargs['bottom'])

	if show_legend:
		ax.legend(**legend)
		
	ax.set_title(title)

	ax.grid(**grid)

	if fig_size is not None:
		w = fig_size[0]/2.54
		h = fig_size[1]/2.54
		fig.set_size_inches((w,h))

	if xscale == 'log':
		ax.xaxis.set_major_formatter(FuncFormatter(format_func))
		if log10(xmax/xmin) < 1:
			ax.xaxis.set_minor_formatter(FuncFormatter(format_func))
		# Note: in Matplotlib version 3.3.4, the following call should be valid
		#ax.xaxis.set_major_formatter(format_func)
	if yscale == 'log':
		ax.yaxis.set_major_formatter(FuncFormatter(format_func))
		# Note: in Matplotlib version 3.3.4, the following call should be valid
		#ax.yaxis.set_major_formatter(format_func)

	if len(filename) > 0:
		plt.savefig(filename, dpi=dpi, bbox_inches='tight',
			  transparent=transparent)

	plt.show()

@set_module('qtools')
def plotps(*args, **kwargs):
	"""Function for plotting instances of class PowerSpectrum and
	FourierSpectrum.

	Parameters
	----------
	*args
		Any number of power spectra or Fourier spectra. All spectra must be
		of the same type.
	**kwargs
		Optional named parameters (see below under **Other parameters**)

	Other parameters
	----------------
	style : str
		Plot style. See `Matplotlib style sheets reference <https://
		matplotlib.org/stable/gallery/style_sheets/
		style_sheets_reference.html>`_ for available styles. Default `default`.
	xscale : {'log', 'lin'}
		Specifies the scale on the x-axis (logarithmic or linear).
		Default 'lin'.
	legend : dict (or bool)
		The keys in this dictionary will be used to set the arguments in a
		call to :func:`matplotlib.pyplot.legend`. If this parameter is not
		provided, the legend will not be shown. This parameter can also be set
		to True to show the legend with default parameters ``{'loc': 'best'}``
	filename : str
		If given, save plot to file using `filename` as file name. The file
		name should include the desired extension (e.g. 'png' or 'svg'), from
		which the file format will be determined as per
		:func:`matplotlib.pyplot.savefig`.
	dpi : int
		Dots per inch to use if plot is saved to file. Default None.
	right : float
		Sets the upper limit on the x-axis.
	left : float
		Sets the lower limit on the x-axis.
	top : float
		Sets the upper limit on the y-axis.
	bottom : float
		Sets the lower limit on the y-axis.
	grid : dict
		The keys in this dictionary will be used to set the arguments in a
		call to :func:`matplotlib.pyplot.grid`. Default: ``{'which': 'major',
		'color': '0.75'}``
	title : str
		Title to be placed above plot.

	Notes
	-----
	The line format can be set in the 'fmt' attribute of a spectrum.
	The line format is passed directly to :func:`matplotlib.pyplot.plot` as
	'fmt'.

	Examples
	--------
	See :func:`qtools.plotrs`.
	"""

	style = kwargs.get('style', 'default')
	xscale = kwargs.get('xscale', 'lin')
	if 'legend' in kwargs:
		show_legend = True
		legend = {'loc': 'best'}
		if isinstance(kwargs['legend'], dict):
			legend = kwargs['legend']
	else:
		show_legend = False
	filename = kwargs.get('filename', '')
	dpi = kwargs.get('dpi', None)
	grid = kwargs.get('grid', {'which': 'major', 'color': '0.75'})
	title = kwargs.get('title', None)

	plt.style.use(style)

	# Get the axes of a new plot
	fig, ax = plt.subplots()

	# Set the appropriate type of plot
	if xscale == 'log':
		plot = ax.semilogx
	elif xscale == 'lin':
		plot = ax.plot

	ax.set_xlabel('Frequency [Hz]')

	# Check the validity of the arguments and set the labels on the y-axis
	if all([isinstance(ps, FourierSpectrum) for ps in args]):
		ax.set_ylabel('Fourier amplitude [{}]'.format(args[0].unit))
		yaxis = 'X'
	elif all([isinstance(ps, PowerSpectrum) for ps in args]):
		ax.set_ylabel('Spectral power density [{}]'.format(args[0].unit))
		yaxis = 'Wf'
	else:
		raise ValueError('The arguments provided to plotps are not all of the'
				   ' same type (either FourierSpectrum or PowerSpectrum)')
	if len(set([ps.unit for ps in args])) > 1:
		Info.warn('In plotps, it seems that some spectra have different '
			'units.')

	# Create the plots
	for ps in args:
		if yaxis=='X':
			y = ps.abs()
		else:
			y = ps.Wf
		if ps.fmt == '_default_':
			plot(ps.f, y, label=ps.label)
		else:
			plot(ps.f, y, ps.fmt, label=ps.label)

	# Set upper limit on x-axis if specified
	if 'right' in kwargs:
		ax.set_xlim(right=kwargs['right'])

	# Set lower limit on x-axis if specified
	if 'left' in kwargs:
		ax.set_xlim(left=kwargs['left'])

	# Set upper limit on y-axis if specified
	if 'top' in kwargs:
		ax.set_ylim(top=kwargs['top'])

	# Set lower limit on y-axis if specified
	if 'bottom' in kwargs:
		ax.set_ylim(bottom=kwargs['bottom'])

	if show_legend:
		ax.legend(**legend)

	ax.set_title(title)
	
	ax.grid(**grid)

	if len(filename) > 0:
		plt.savefig(filename, dpi=dpi, bbox_inches='tight')

	plt.show()

@set_module('qtools')
def plotth(*args, **kwargs):
	"""Function for plotting instances of class TimeHistory.

	Parameters
	----------
	*args
		Any number of time histories
	**kwargs
		Optional named parameters (see below under **Other parameters**)

	Other parameters
	----------------
	bottom : float
		Sets the lower limit on the y-axis.
	dpi : int
		Dots per inch to use if plot is saved to file. Default None.
	filename : str
		If given, save plot to file using `filename` as file name. The file
		name should include the desired extension (e.g. 'png' or 'svg'), from
		which the file format will be determined as per
		:func:`matplotlib.pyplot.savefig`.
	grid : dict
		The keys in this dictionary will be used to set the arguments in a
		call to :func:`matplotlib.pyplot.grid`. Default: ``{'which': 'major',
		'color': '0.75'}``
	left : float
		Sets the lower limit on the x-axis.
	legend : bool
		Display the legend on the plot. Default False.
	right : float
		Sets the upper limit on the x-axis.
	title : str
		Title to be placed above plot.
	top : float
		Sets the upper limit on the y-axis.

	Notes
	-----
	The line format can be set in the 'fmt' attribute of a response spectrum.
	The line format is passed directly to :func:`matplotlib.pyplot.plot` as
	'fmt'.
	"""
	xscale = kwargs.get('xscale','lin')
	show_legend = kwargs.get('legend',False)
	filename = kwargs.get('filename','')
	dpi = kwargs.get('dpi',None)
	title = kwargs.get('title', None)
	grid = kwargs.get('grid', {'which': 'major', 'color': '0.75'})

	plt.xlabel('Time [s]')

	if args[0].ordinate == 'a':
		plt.ylabel('Acceleration')
	elif args[0].ordinate == 'v':
		plt.ylabel('Velocity')
	elif args[0].ordinate == 'd':
		plt.ylabel('Displacement')

	for th in args:
		if xscale == 'log':
			plt.semilogx(th.time, th.data, th.fmt, label=th.label)
		elif xscale == 'lin':
			plt.plot(th.time, th.data, th.fmt, label=th.label)

	# Set upper limit on x-axis if specified
	if 'right' in kwargs:
		plt.xlim(right=kwargs['right'])

	# Set lower limit on x-axis if specified
	if 'left' in kwargs:
		plt.xlim(left=kwargs['left'])

	# Set upper limit on y-axis if specified
	if 'top' in kwargs:
		plt.ylim(top=kwargs['top'])

	# Set lower limit on y-axis if specified
	if 'bottom' in kwargs:
		plt.ylim(bottom=kwargs['bottom'])

	if show_legend:
		plt.legend(loc='best')
	
	plt.title(title)
	
	plt.grid(**grid)

	if len(filename) > 0:
		plt.savefig(filename, dpi=dpi)

	plt.show()