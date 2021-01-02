"""
Package: Qtools
Module: systems
(C) 2020-2021 Andreas H. Nielsen
See README.md for further details.
"""

from math import sin, sqrt
import numpy as np

class Bilinear:
	"""
	Class for a bilinear 1DOF system with elastic stiffness k,
	mass m, and postyield stiffness ratio r.
	The class implements a kinematic hardening rule.
	"""

	def __init__(self,m,k,r,beta,Fy):

		self.m = m
		self.k = k
		self.H = r*k/(1-r)
		self.q = 0
		self.x0 = 0
		self.xp = 0
		self.Fs = 0
		self.beta = beta
		self.Fy = Fy
		self.xsol = []

	def __call__(self,x1,t,th):

		# Retrieve state parameters
		x0 = self.x0
		Fs = self.Fs
		q = self.q
		xp = self.xp

		# Elastic predictor
		dx = x1[0]-x0
		Fs = Fs + self.k*dx

		# Update state parameters
		self.x0 = x1[0]
		if abs(Fs-q)-self.Fy <= 0:
			self.Fs = Fs
		else:
			dxp = (abs(Fs-q)-self.Fy)/(self.k+self.H)
			self.Fs = Fs - np.sign(Fs-q)*self.k*dxp
			self.q = q + np.sign(Fs-q)*self.H*dxp
			self.xp = xp + dxp

		s = th(t)
		f = np.array([x1[1], -s-self.Fs/self.m])
		return f

	def evalFs(self,xsol,verbose=False):

		# local state parameters
		q = 0
		xp = 0

		N = np.size(xsol)
		Fs = np.zeros(N)

		for i in range(N-1):

			# Retrieve solution
			x0 = xsol[i]
			x1 = xsol[i+1]

			# Elastic predictor
			dx = x1-x0
			Fstr = Fs[i] + self.k*dx

			# Update state parameters
			if abs(Fstr-q)-self.Fy <= 0:
				Fs[i+1] = Fstr
			else:
				dxp = (abs(Fstr-q)-self.Fy)/(self.k+self.H)
				Fs[i+1] = Fstr - np.sign(Fstr-q)*self.k*dxp
				q = q + np.sign(Fstr-q)*self.H*dxp
				xp = xp + dxp

		if verbose:
			print('End force comparison: post-processed: {:.2f}, state variable: {:.2f}'.fomuat(Fs[-1],self.Fs))
			print('Centre of elastic domain: post-processed: {:.2f}, state variable: {:.2f}'.fomuat(q,self.q))
			print('Accumulated plastic strain: post-processed: {:.2f}, state variable: {:.2f}'.fomuat(xp,self.xp))

		return Fs

def vis1(x,t,omega,xi,th):
	"""
	Viscous damping 1DOF model
	Used by odeint solver
	"""
	# x[0] is displacement
	# x[1] is velocity
	# f[0] is velocity
	# f[1] is acceleration

	s = th(t)
	f = np.array([x[1], -s-2*xi*omega*x[1]-omega**2*x[0]])
	return f

def sld(x,t,omega,xi,th):
	"""
	'Modified solid' damping 1DOF model
	"""

	bm = 5.
	s = th(t)
	f = np.array([x[1], -s-bm*omega**2*x[1]*abs(x[0])-omega**2*x[0]])

	return f

def hsld(x,t,m,b,ud,k,F0,wf):
	"""
	'Modified solid' damping 1DOF model with harmonic loading
	"""

	f = np.array([x[1], (F0*sin(wf*t)-b*x[1]*(abs(x[0])+ud)-k*x[0])/m])

	return f

def ndof(x,t,M,C,K,th):
	"""
	Generalised N-dof system with viscuos damping.
	Assumes a diagonal mass matrix.
	"""

	s = th(t)
	m = x.size
	n = m//2
	f = np.empty(m)
	u = x[0:n]  # Displacement
	v = x[n:m]  # Velocity
	f[0:n] = x[n:m]
	Mi = 1/np.diagonal(M)
	f[n:m] = -s-Mi*(np.dot(C,v)+np.dot(K,u))

	return f

def DOF_modal(mu,rhos,xi):

	rho1 = sqrt((1+(1+mu)*rhos**2-sqrt((1+(1+mu)*rhos**2)**2-4*rhos**2))/2)
	rho2 = sqrt((1+(1+mu)*rhos**2+sqrt((1+(1+mu)*rhos**2)**2-4*rhos**2))/2)
	phi1 = np.array([1-(rho1/rhos)**2,1])
	phi2 = np.array([1-(rho2/rhos)**2,1])
	gam1 = (1-(rho1/rhos)**2+mu)/((1-(rho1/rhos)**2)**2+mu)
	gam2 = (1-(rho2/rhos)**2+mu)/((1-(rho2/rhos)**2)**2+mu)
	return ((rho1,rho2),(phi1,phi2),(gam1,gam2))


