# -*- coding: utf-8 -*-
"""
This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as tr
from pylab import *
from netCDF4 import Dataset
from cmath import *
import csv  
#import math 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

import MPAinterpolationAuto as mpa
import MPAsampAuto as samp

import tools as ts

# Functions ------------------------------------------------------------
def mpa_R_fit(nfreq, npols, w, x, E):
    # Transforming the problem into a 2* larger least square with real numbers:
    A = np.zeros((nfreq*2,npols*2),dtype='complex64')
    b = np.zeros((nfreq*2),dtype='complex64')
    for k in range(nfreq):
      b[2*k]   = np.real(x[k])
      b[2*k+1] = np.imag(x[k])
      for i in range(npols):
        A[2*k  ][2*i  ] =  np.real(2*E[i]/(w[k]**2 -E[i]**2)) 
        A[2*k  ][2*i+1] = -np.imag(2*E[i]/(w[k]**2 -E[i]**2))
        A[2*k+1][2*i  ] =  np.imag(2*E[i]/(w[k]**2 -E[i]**2)) 
        A[2*k+1][2*i+1] =  np.real(2*E[i]/(w[k]**2 -E[i]**2))

    Rri = np.linalg.lstsq(A, b, rcond=None)[0]

    R = np.zeros(npols,dtype='complex64')
    R = Rri[::2] + 1j*Rri[1::2]

    return R

def mpa_rR_fit(nfreq, npols, w, b, rE, iE):
    A = np.zeros((nfreq,npols),dtype='complex64')
    for k in range(nfreq):
      for p in range(npols):
        A[k][p] =  iE[p]/(iE[p]**2 + (w[k] - rE[p])**2) 

    rR = np.linalg.lstsq(A, b, rcond=None)[0]

    return rR


# General variables ----------------------------------------------

path='../Cr_k36/'
out_ip_q='o-opt-IP_b500_x1RL_r0-200eV_d0.1eV_f2000.'
out_ha_q='o-opt-HA_b500_x5Ry_r0-200eV_d0.1eV_f2000.'

eps='eps_q'
eel='eel_q'
theoIP='_ip'
theoHA='_inv_rpa_dyson'

q = 5
ener,eel_i,eel_r = np.genfromtxt(path+out_ha_q+eel+str(q)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
#eel_i=eel_i*ener**2; eel_r=eel_r*ener**2
maxfreq = ener[-1]#*0.95
egrid, eel_r_interp = ts.interpolate_spectrum(ener,eel_r,maxfreq,0)
egrid, eel_i_interp = ts.interpolate_spectrum(ener,eel_i,maxfreq,0)
eel_i_denoised = ts.denoise_spectrum(eel_i_interp,ws=101)
eel_i_findpols = eel_i_interp

from scipy.signal import argrelextrema, find_peaks, peak_widths, peak_prominences
# for local maxima
maxims = argrelextrema(eel_i_denoised, np.greater)
print('number of local maxima',len(maxims[0]))

# for local minima
minims = argrelextrema(eel_i_findpols, np.less)

realpeaks, _ = find_peaks(eel_i_findpols)
realpeaks = maxims[0]
#print('ener peaks',egrid[realpeaks]) 
resipeaks = peak_prominences(eel_i_findpols, realpeaks)
#print('resi peaks',resipeaks[0]) 
imagpeaks_half = peak_widths(eel_i_findpols, realpeaks, rel_height=0.5, prominence_data=resipeaks)
imagpeaks_full = peak_widths(eel_i_findpols, realpeaks, rel_height=1, prominence_data=resipeaks)
imagpeaks_0p = peak_widths(eel_i_findpols, realpeaks, rel_height=0.3, prominence_data=resipeaks)
#print('half width peaks',imagpeaks_half[0]*egrid[-1]/len(eel_i_denoised))
#print('full width peaks',imagpeaks_full[0]*egrid[-1]/len(eel_i_denoised))

rE = egrid[realpeaks]
#iE = imagpeaks_half[0]*egrid[-1]/len(eel_i_denoised)
#iE = imagpeaks_0p[0]*ener[-1]/len(eel_i_findpols)
iE = rE
rE0 = np.concatenate(([egrid[0]],rE,[egrid[-1]]))
for p in range(1,len(rE0)-2):
  iE[p] = (rE0[p-1]+rE0[p+1])*0.3

iz = 0.1
npols = len(realpeaks)
E = rE - 1j*iE
"""nfreq = len(ener)
w = ener + 1j*iz
x = eel_r - 1j*eel_i
print('E',E)"""
nfreq = len(egrid)
w = egrid + 1j*iz
x = eel_r_interp- 1j*eel_i_interp

R = mpa_R_fit(nfreq, npols, w, x, E)
rR = mpa_rR_fit(nfreq, npols, egrid, eel_i_interp, rE, iE)
Xfit = np.zeros_like(x)
iXfit = np.zeros_like(x)
for p in range(npols):
  Xfit = Xfit + 2*R[p]*E[p]/(w**2 - E[p]**2)
  iXfit = iXfit + rR[p]*iE[p]/(iE[p]**2 + (egrid - rE[p])**2)




"""scale = max(eel_i_denoised)
precision = 0.009
for i in maxims[0]:
  if eel_i_denoised[i]/scale > precision:
    print(egrid[i],eel_i_denoised[i])

zipped_lists = zip(eel_i_denoised[maxims[0]], egrid[maxims[0]])
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
x, rE = [list(tuple) for tuple in tuples]

npols = 5
rE = np.array(rE[-npols:])
print(npols, rE)
iz = 0.1
from scipy.optimize import curve_fit
def P3 (rz, R0, R1, R2, iE0, iE1, iE2):
    return -R0*(iE0-iz)/((rz-rE[0])**2+(iE0-iz)**2) \
           -R1*(iE1-iz)/((rz-rE[1])**2+(iE1-iz)**2) \
           -R2*(iE2-iz)/((rz-rE[2])**2+(iE2-iz)**2)
def P5 (rz, R0, R1, R2, R3, R4, iE0, iE1, iE2, iE3, iE4):
    return -R0*(iE0-iz)/((rz-rE[0])**2+(iE0-iz)**2) \
           -R1*(iE1-iz)/((rz-rE[1])**2+(iE1-iz)**2) \
           -R2*(iE2-iz)/((rz-rE[2])**2+(iE2-iz)**2) \
           -R3*(iE3-iz)/((rz-rE[3])**2+(iE3-iz)**2) \
           -R4*(iE4-iz)/((rz-rE[4])**2+(iE4-iz)**2)
def P9 (rz, R0, R1, R2, R3, R4, R5, R6, R7, R8, iE0, iE1, iE2, iE3, iE4, iE5, iE6, iE7, iE8):
    return -R0*(iE0-iz)/((rz-rE[0])**2+(iE0-iz)**2) \
           -R1*(iE1-iz)/((rz-rE[1])**2+(iE1-iz)**2) \
           -R2*(iE2-iz)/((rz-rE[2])**2+(iE2-iz)**2) \
           -R3*(iE3-iz)/((rz-rE[3])**2+(iE3-iz)**2) \
           -R4*(iE4-iz)/((rz-rE[4])**2+(iE4-iz)**2) \
           -R5*(iE4-iz)/((rz-rE[5])**2+(iE5-iz)**2) \
           -R6*(iE4-iz)/((rz-rE[6])**2+(iE6-iz)**2) \
           -R7*(iE4-iz)/((rz-rE[7])**2+(iE7-iz)**2) \
           -R8*(iE4-iz)/((rz-rE[8])**2+(iE8-iz)**2) 
def P10 (rz, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, iE0, iE1, iE2, iE3, iE4, iE5, iE6, iE7, iE8, iE9):
    return -R0*(iE0-iz)/((rz-rE[0])**2+(iE0-iz)**2) \
           -R1*(iE1-iz)/((rz-rE[1])**2+(iE1-iz)**2) \
           -R2*(iE2-iz)/((rz-rE[2])**2+(iE2-iz)**2) \
           -R3*(iE3-iz)/((rz-rE[3])**2+(iE3-iz)**2) \
           -R4*(iE4-iz)/((rz-rE[4])**2+(iE4-iz)**2) \
           -R5*(iE4-iz)/((rz-rE[5])**2+(iE5-iz)**2) \
           -R6*(iE4-iz)/((rz-rE[6])**2+(iE6-iz)**2) \
           -R7*(iE4-iz)/((rz-rE[7])**2+(iE7-iz)**2) \
           -R8*(iE4-iz)/((rz-rE[8])**2+(iE8-iz)**2) \
           -R9*(iE4-iz)/((rz-rE[9])**2+(iE9-iz)**2)

# Visualize the results
fx=1;fy=1
fig2, axs2 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig2.set_size_inches(f*4*fx, f*3*fy)
axs2.scatter(egrid, eel_i_denoised, label='Data')

func = P5
popt, pcov = curve_fit(func, egrid, eel_i_denoised)
axs2.plot(egrid, func(egrid, *popt), 'r-', label='Fit')
axs2.legend()"""

# Plots ----------------------------------------------------------
plt.rcParams['font.size'] = '11.5'
plt.rcParams['lines.linewidth'] = 1.5

f=0.9
fx=1;fy=2
fig1, axs1 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig1.set_size_inches(f*4*fx, f*3*fy)

axs1[0].plot(ener, eel_r)
axs1[1].plot(ener, eel_i)
axs1[1].plot(egrid, eel_i_interp)
axs1[1].plot(egrid, eel_i_denoised)

# Visualize the results
fx=1;fy=1
fig2, axs2 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig2.set_size_inches(f*4*fx, f*3*fy)
axs2.scatter(egrid, eel_i_denoised, label='Data')

#axs1[0].plot(ener, np.real(Xfit))
#axs1[1].plot(ener, np.imag(-Xfit))
axs1[1].plot(egrid, iXfit)

######
plt.show()
