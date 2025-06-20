"""
Package: Qtools
Module: config
(C) 2020-2025 Andreas H. Nielsen
See README.md for further details.
"""

# Configuration parameters
version = '3.3'
# Note to self: remember to also update version in the following files:
# ./README.md

# Decorator
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

# Define a class that holds global information setttings
@set_module('qtools')
class Info:
	"""
	Provides output from Qtools.

	Notes
	-----
	Use ``Info.setLevel`` to set the level of output. For example::

		>>> qt.Info.setLevel(1)

	The following output levels are recognised:

		0. No output (silent mode)
		1. Warnings only
		2. Warnings and general information (default)
		3. All of the above and debug information
	"""

	# Set default level
	level = 2

	@classmethod
	def setLevel(cls, lv):
		"""Sets the output level."""
		cls.level = lv

	@classmethod
	def getLevel(cls):
		"""Returns the current output level."""
		return cls.level

	@classmethod
	def warn(cls, warning, prepend=True):
		if cls.level >= 1:
			if prepend:
				warning = 'WARNING: ' + warning
			print(warning)

	@classmethod
	def note(cls, info):
		if cls.level >= 2:
			print(info)

	@classmethod
	def deb(cls, info, prepend=True):
		if cls.level >= 3:
			if prepend:
				info = 'DEBUG: ' + info
			print(info)

	@classmethod
	def end(cls):
		if cls.level >= 2:
			print('------------------------------------------')
