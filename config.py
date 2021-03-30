"""
Package: Qtools
Module: config
(C) 2020-2021 Andreas H. Nielsen
See README.md for further details.
"""

# Configuration parameters
version = '2.2'
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
