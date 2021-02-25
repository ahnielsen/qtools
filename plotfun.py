"""
Package: Qtools
Module: plotfun
(C) 2020-2021 Andreas H. Nielsen
See README.md for further details.
"""

import matplotlib.pyplot as plt
from qtools import config
from qtools.power_spectrum import PowerSpectrum, FourierSpectrum

@config.set_module('qtools')
def plotrs(*args,**kwargs):
	"""Function for plotting instances of class ResponseSpectrum.

	Parameters
	----------
	*args
		Any number of response spectra
	**kwargs
		Optional parameters (see below under **Other parameters**)

	Other parameters
	----------------
	xaxis : {'f', 'T'}
		Quantity to plot on the x-axis. Default 'f'.
	yaxis : {'sa', 'sv', 'sd'}
		Quantity to plot on the y-axis. Default 'sa'.
	xscale : {'log', 'lin'}
		Specifies the scale on the x-axis (logarithmic or linear).
		Default 'log'.
	legend : bool
		Display the legend on the plot. Default False.
	filename : str
		If given, save plot to file using `filename` as file name. The file
		name should include the desired extension (e.g. 'png' or 'svg'), from
		which the file format will be determined as per
		:func:`matplotlib.pyplot.savefig`.
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

	Notes
	-----
	The line format can be set in the 'fmt' attribute of a response spectrum.
	The line format is passed directly to :func:`matplotlib.pyplot.plot` as
	'fmt'.

	Examples
	--------
	A typical call with 3 response spectra and default options is::

		qt.plotrs(rs1,rs2,rs3)

	If the spectra are contained in a list (say, `rslist`), unpack the list
	inside the call using the star operator::

		qt.plotrs(*rslist)

	"""
	# Get the parameters
	xaxis = kwargs.get('xaxis','f')
	yaxis = kwargs.get('yaxis','sa')
	xscale = kwargs.get('xscale','log')
	show_legend = kwargs.get('legend',False)
	filename = kwargs.get('filename','')
	dpi = kwargs.get('dpi',None)

	# Set the appropriate type of plot
	if xscale == 'log':
		plot = plt.semilogx
	elif xscale == 'lin':
		plot = plt.plot

	# Set the labels on the axes
	if xaxis == 'f':
		plt.xlabel('Frequency [Hz]')
	elif xaxis == 'T':
		plt.xlabel('Period [s]')
	else:
		raise ValueError('{} is not a valid value for the x-axis on a'
				   ' response spectrum plot'.format(xaxis))
	if yaxis == 'sa':
		plt.ylabel('Spectral acceleration [g]')
	elif yaxis == 'sv':
		plt.ylabel('Spectral velocity [m/s]')
	elif yaxis == 'sd':
		plt.ylabel('Spectral displacement [m]')
	elif yaxis == 'ei':
		plt.ylabel('Spectral input energy [J/kg]')
	else:
		raise ValueError('{} is not a valid value for the y-axis on a'
				   ' response spectrum plot'.format(yaxis))

	for rs in args:
		x = rs.__dict__[xaxis]
		y = rs.__dict__[yaxis]
		if rs.color == '':
			plot(x, y, rs.fmt, label=rs.label)
		else:
			plot(x, y, rs.fmt, label=rs.label, color=rs.color)

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

	plt.grid(color='0.75')

	if len(filename) > 0:
		plt.savefig(filename, dpi=dpi)

	plt.show()

@config.set_module('qtools')
def plotps(*args, **kwargs):
	"""Function for plotting instances of class PowerSpectrum and
	FourierSpectrum.

	Parameters
	----------
	*args
		Any number of power spectra or Fourier spectra. All spectra must be
		of the same type.
	**kwargs
		Optional parameters (see below under **Other parameters**)

	Other parameters
	----------------
	xscale : {'log', 'lin'}
		Specifies the scale on the x-axis (logarithmic or linear).
		Default 'lin'.
	legend : bool
		Display the legend on the plot. Default False.
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

	Examples
	--------
	See :func:`qtools.plotrs`.
	"""

	xscale = kwargs.get('xscale', 'lin')
	show_legend = kwargs.get('legend', False)
	filename = kwargs.get('filename', '')
	dpi = kwargs.get('dpi', None)

	# Set the appropriate type of plot
	if xscale == 'log':
		plot = plt.semilogx
	elif xscale == 'lin':
		plot = plt.plot

	plt.xlabel('Frequency [Hz]')

	# Check the validity of the arguments and set the labels on the y-axis
	if all([isinstance(ps, FourierSpectrum) for ps in args]):
		plt.ylabel('Fourier amplitude [{}]'.format(args[0].unit))
		yaxis = 'X'
	elif all([isinstance(ps, PowerSpectrum) for ps in args]):
		plt.ylabel('Spectral power density [{}]'.format(args[0].unit))
		yaxis = 'Wf'
	else:
		raise ValueError('The arguments provided to plotps are not all of the'
				   ' same type (either FourierSpectrum or PowerSpectrum)')
	if len(set([ps.unit for ps in args])) > 1:
		config.vprint('WARNING: in plotps, it seems that some spectra have'
			 ' different units.')

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
	
	plt.grid(color='0.75')

	if len(filename) > 0:
		plt.savefig(filename, dpi=dpi)

	plt.show()

@config.set_module('qtools')
def plotth(*args,**kwargs):
	"""Function for plotting instances of class TimeHistory.

	Parameters
	----------
	*args
		Any number of time histories
	**kwargs
		Optional parameters (see below under **Other parameters**)

	Other parameters
	----------------
	legend : bool
		Display the legend on the plot. Default False.
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

	if len(filename) > 0:
		plt.savefig(filename,dpi=dpi)

	plt.show()
