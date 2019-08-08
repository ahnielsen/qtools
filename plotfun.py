"""
Module qtools.plotfun
Version 0.8 (beta)
Depends on: Python 3.4, Numpy 1.10
Contains functions for plotting.
@author: Andreas H Nielsen
"""

import matplotlib.pyplot as plt
import numpy as np
from math import floor

def plotrs(*args,**kwargs):
    """Function for plotting response spectra of class ResponseSpectrum.
    
    *args = any number of response spectra
    
    **kwargs options:
    xaxis = ['f|'T'] (default 'f');
    yaxis = ['sa','sv','sd'] (default 'sa');
    xscale = ['log'|'lin'] (default 'log');
    legend = [True|False] (default False);
    save_fig = [True|False] (default False);
    right = %f, set the upper limit on the x-axis;
    left = %f, set the lower limit on the x-axis;
    """
    
    xaxis = kwargs.get('xaxis','f')
    yaxis = kwargs.get('yaxis','sa')
    if yaxis not in ('sa','sv','sd'):
        raise ValueError('The parameter yaxis must be either \'sa\', \'sv\' or \'sd\'.')
    xscale = kwargs.get('xscale','log')
    show_legend = kwargs.get('legend',False)
    save_fig = kwargs.get('save_fig',False)
    
    if xaxis == 'f':
        plt.xlabel('Frequency [Hz]')
        if yaxis == 'sa':
            plt.ylabel('Spectral acceleration [g]')
            ymax = 0.0
            for rs in args:
                ymax = max(np.amax(rs.sa),ymax)
                if xscale == 'log':
                    plt.semilogx(rs.f,rs.sa,label=rs.label)
                elif xscale == 'lin':
                    plt.plot(rs.f,rs.sa,label=rs.label)
    
    if xaxis == 'T':
        plt.xlabel('Period [s]')
        if yaxis == 'sa':
            plt.ylabel('Spectral acceleration [g]')
            ymax = 0.0
            for rs in args:
                ymax = max(np.amax(rs.sa),ymax)
                if xscale == 'log':
                    plt.semilogx(rs.T,rs.sa,label=rs.label)
                elif xscale == 'lin':
                    plt.plot(rs.T,rs.sa,label=rs.label)
    
    if xaxis == 'f':
        plt.xlabel('Frequency [Hz]')
        if yaxis == 'sv':
            plt.ylabel('Spectral velocity [m/s]')
            ymax = 0.0
            for rs in args:
                ymax = max(np.amax(rs.sv),ymax)
                if xscale == 'log':
                    plt.semilogx(rs.f,rs.sv,label=rs.label)
                elif xscale == 'lin':
                    plt.plot(rs.f,rs.sv,label=rs.label)
    
    if xaxis == 'T':
        plt.xlabel('Period [s]')
        if yaxis == 'sv':
            plt.ylabel('Spectral velocity [m/s]')
            ymax = 0.0
            for rs in args:
                ymax = max(np.amax(rs.sv),ymax)
                if xscale == 'log':
                    plt.semilogx(rs.T,rs.sv,label=rs.label)
                elif xscale == 'lin':
                    plt.plot(rs.T,rs.sv,label=rs.label)
    
    if xaxis == 'f':
        plt.xlabel('Frequency [Hz]')
        if yaxis == 'sd':
            plt.ylabel('Spectral displacement [m]')
            ymax = 0.0
            for rs in args:
                ymax = max(np.amax(rs.sd),ymax)
                if xscale == 'log':
                    plt.semilogx(rs.f,rs.sd,label=rs.label)
                elif xscale == 'lin':
                    plt.plot(rs.f,rs.sd,label=rs.label)
    
    if xaxis == 'T':
        plt.xlabel('Period [s]')
        if yaxis == 'sd':
            plt.ylabel('Spectral displacement [m]')
            ymax = 0.0
            for rs in args:
                ymax = max(np.amax(rs.sd),ymax)
                if xscale == 'log':
                    plt.semilogx(rs.T,rs.sd,label=rs.label)
                elif xscale == 'lin':
                    plt.plot(rs.T,rs.sd,label=rs.label)

    # Check the upper limit and increase if necessary
    cur_ymin, cur_ymax = plt.ylim()
    ymax = float(floor(int(10*ymax+1)))/10.
    if cur_ymax < ymax:
        plt.ylim((0.0,ymax))
        
    # Set upper limit on x-axis if specified
    if 'right' in kwargs:
        if type(kwargs['right']) is float:
            plt.xlim(right=kwargs['right'])

    # Set lower limit on x-axis if specified
    if 'left' in kwargs:
        if type(kwargs['left']) is float:
            plt.xlim(left=kwargs['left'])
            
    if show_legend:
        plt.legend(loc='best')

    plt.grid(color='0.75')

    if save_fig:
        plt.savefig('rsplot_save.png',dpi=1000,format='png')

    plt.show()
    
    
def plotps(*args,**kwargs):
    """Function for plotting power spectra of class PowerSpectrum.
    
    *args = any number of power spectra
    
    **kwargs options:
    xaxis = ['f|'T'|'w'] (default 'f');
    yaxis = ['Sw','Sk','X'] (default 'Sw');
    xscale = ['log'|'lin'] (default 'lin');
    legend = [True|False] (default True).
    """
    
    xaxis = kwargs.get('xaxis','f')
    yaxis = kwargs.get('yaxis','Sw')
    xscale = kwargs.get('xscale','lin')
    show_legend = kwargs.get('legend',False)
    
    if xaxis == 'f':
        plt.xlabel('Frequency [Hz]')
        if yaxis == 'Sw':
            plt.ylabel('Spectral power density')
            ymax = 0.0
            for ps in args:
                ymax = max(np.amax(ps.Sw),ymax)
                if xscale == 'log':
                    plt.semilogx(ps.f,ps.Sw,label=ps.label)
                elif xscale == 'lin':
                    plt.plot(ps.f,ps.Sw,label=ps.label)
    
    if xaxis == 'f':
        plt.xlabel('Frequency [Hz]')
        if yaxis == 'X':
            plt.ylabel('Amplitude [unit]')
            ymax = 0.0
            for ps in args:
                ymax = max(np.amax(ps.X),ymax)
                if xscale == 'log':
                    plt.semilogx(ps.f,ps.X,label=ps.label)
                elif xscale == 'lin':
                    plt.plot(ps.f,ps.X,label=ps.label)
    
    if xaxis == 'f':
        plt.xlabel('Frequency [Hz]')
        if yaxis == 'Sk0':
            plt.ylabel('Amplitude [unit]')
            ymax = 0.0
            for ps in args:
                ymax = max(np.amax(ps.Sk),ymax)
                if xscale == 'log':
                    plt.semilogx(ps.f,ps.Sk0,label=ps.label)
                elif xscale == 'lin':
                    plt.plot(ps.f,ps.Sk0,label=ps.label)

    # Set right if specified
    if 'right' in kwargs:
        targ = type(kwargs['right']) 
        if (targ is float) or (targ is int):
            plt.xlim(right=kwargs['right'])

    # Set left if specified
    if 'left' in kwargs:
        targ = type(kwargs['left'])
        if (targ is float) or (targ is int):
            plt.xlim(left=kwargs['left'])
            
    if show_legend:
        plt.legend(loc='best')

    plt.show()

def plotth(*args,**kwargs):
    
    xscale = kwargs.get('xscale','lin')
    show_legend = kwargs.get('legend',False)

    plt.xlabel('Time [s]')
    
    if args[0].ordinate == 'a':
        plt.ylabel('Acceleration')
    elif args[0].ordinate == 'v':
        plt.ylabel('Velocity')
    elif args[0].ordinate == 'd':
        plt.ylabel('Displacement')

    for th in args:
        if xscale == 'log':
            plt.semilogx(th.time,th.data,label=th.label)
        elif xscale == 'lin':
            plt.plot(th.time,th.data,label=th.label)  
    
    # Set upper limit on x-axis if specified
    if 'right' in kwargs:
        if type(kwargs['right']) is float:
            plt.xlim(right=kwargs['right'])

    # Set lower limit on x-axis if specified
    if 'left' in kwargs:
        if type(kwargs['left']) is float:
            plt.xlim(left=kwargs['left'])
            
    if show_legend:
        plt.legend(loc='best')

    plt.show()
