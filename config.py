#from scipy.interpolate import interp1d as si1d
#import numpy as np
#from math import nan

# Configuration parameters
version = '0.2'
verbose = True

# Define function that will print output if verbose
vprint = print if verbose else lambda *a, **k: None

def set_module(module):
	"""
	Decorator for overriding __module__ on a function or class.

	Example usage::

		@set_module('numpy')
		def example():
			pass

		assert example.__module__ == 'numpy'
	"""
	def decorator(func):
		if module is not None:
			func.__module__ = module
			return func
	
	return decorator

# Define a class that is a subclass of scipy.interpolate.interp1d
# NOTE: this is currently not used anywhere within Qtools
#class interpol(si1d):
#
#	def __init__(self, x, y, kind='linear', axis=-1, copy=True, bounds_error=None,
#			  fill_value=nan, assume_sorted=False):
#
#		self.kind2 = kind
#
#		if kind == 'loglin':
#			if np.less_equal(x,0).any():
#				raise ValueError('All values in x must be greater than zero with interpolation kind = \'loglin\'')
#			si1d.__init__(self, np.log(x), y, kind='linear', axis=axis, copy=copy,
#				 bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
#		elif kind == 'loglog':
#			if np.less_equal(x,0).any() or np.less_equal(y,0).any():
#				raise ValueError('All values in x and y must be greater than zero with interpolation kind = = \'loglog\'')
#			si1d.__init__(self, np.log(x), np.log(y), kind='linear', axis=axis, copy=copy,
#				 bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
#		elif kind == 'linlog':
#			if np.less_equal(y,0).any():
#				raise ValueError('All values in y must be greater than zero with interpolation kind = = \'linlog\'')
#			si1d.__init__(self, x, np.log(y), kind='linear', axis=axis, copy=copy,
#				 bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
#		else:
#			si1d.__init__(self, x, y, kind=kind, axis=axis, copy=copy,
#				 bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
#
#	def __call__(self,x):
#
#		if self.kind2 == 'loglin':
#			return si1d.__call__(self,np.log(x))
#		elif self.kind2 == 'loglog':
#			return np.exp(si1d.__call__(self,np.log(x)))
#		elif self.kind2 == 'linlog':
#			return np.exp(si1d.__call__(self,x))
#		else:
#			return si1d.__call__(self,x)

