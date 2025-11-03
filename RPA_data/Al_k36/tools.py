# -*- coding: utf-8 -*-
"""
This is a temporary script file.
"""

import numpy as np

#  Kristian
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


# Functions ------------------------------------------------------
def parab(a,b,c,x):

    return a*(x - b)**2 + c


def reX(wr,wi,np,Er,Ei,Rr,Ri):
    reX = 0.
    for i in range(np):
        reX = reX -(2*(Ei[i]**3*Ri[i] +Ei[i]**2*Er[i]*Rr[i] +Ei[i]*(Er[i]**2*Ri[i]
                   -Ri[i]*wi**2 -2*Rr[i]*wi*wr +Ri[i]*wr**2) +Er[i]*(Er[i]**2*Rr[i]
                   +Rr[i]*wi**2 -2*Ri[i]*wi*wr -Rr[i]*wr**2)))/(Ei[i]**4 + Er[i]**4
                   -8*Ei[i]*Er[i]*wi*wr +2*Er[i]**2*(wi**2 -wr**2) +2*Ei[i]**2*
                   (Er[i]**2 -wi**2 + wr**2) +(wi**2 + wr**2)**2)

    return reX


def imX(wr,wi,np,Er,Ei,Rr,Ri):
    imX = 0.
    for i in range(np):
        imX = imX +(2*(Ei[i]**3*Rr[i] -Ei[i]**2*Er[i]*Ri[i] -Er[i]*(Er[i]**2*Ri[i]
                   +Ri[i]*wi**2 +2*Rr[i]*wi*wr -Ri[i]*wr**2) +Ei[i]*(Er[i]**2*Rr[i]
                   -Rr[i]*wi**2 + 2*Ri[i]*wi*wr +Rr[i]*wr**2)))/(Ei[i]**4 +Er[i]**4
                   -8*Ei[i]*Er[i]*wi*wr +2*Er[i]**2*(wi**2 -wr**2) +2*Ei[i]**2*
                   (Er[i]**2 -wi**2 +wr**2) +(wi**2 +wr**2)**2)

    return imX


def reimX1(wr,Iwi,xr,xi):

    a=(xi[0]*xi[1] -xi[1]**2 +xr[1]*(xr[0] -xr[1]))/(
       xi[0]**2 -2*xi[0]*xi[1] +xi[1]**2 +(xr[0] -xr[1])**2)
    b=(xi[1]*xr[0] -xi[0]*xr[1])/(
       xi[0]**2 -2*xi[0]*xi[1] +xi[1]**2 +(xr[0] -xr[1])**2)

    E=Iwi/sqrt(complex(xr[0],xi[0])/complex(xr[1],xi[1]) -1)
    #E=complex(1.0,-0.00001)
    R=-E*complex(xr[0],xi[0])/2
    #R=-(Iwi**2+E**2)*complex(xr[0],xi[0])/(2*E)
    print('E:',E)
    print('R:',R)

    reX1=[]
    imX1=[]
    for i in range(len(wr)):
        X1=2*E*R/(wr[i]**2-E**2)
        reX1.append(X1.real)
        imX1.append(X1.imag)

    return reX1,imX1


def find_plasmon(w_r,eps_r):

    indexes=[]
    plasmons=[]
    for i in range(1,len(w_r)):
      if( eps_r[i-1]*eps_r[i] < 0 ):
        indexes.append(i)
        plasmons.append( (w_r[i-1]*eps_r[i]-w_r[i]*eps_r[i-1] )/(eps_r[i] - eps_r[i-1]) )

    return indexes,plasmons


def normalize_spectrum(ener,eel_i,iq,scale,E0=2.6):

    eel_norm = 2559997.0  # q = 0.2 at ener = 42 eV
    w_e = np.linspace(0, E0, int(E0*50+1))
    e_0 = np.zeros_like(w_e)

    ind = 0
    while ener[-ind] < E0:
        ind += 1
    qfac = (1 + iq / 100) * eel_norm / eel_i[-ind]
    eel_i_normalized = eel_i * scale * qfac

    ener = np.concatenate((w_e,ener[ind:]))
    eel_i_normalized = np.concatenate((e_0,eel_i_normalized[ind:]))

    return ener, eel_i_normalized


def denoise_spectrum(eel_i,ws=31):

    # Denoise data using the Savitzky-Golay filter
    window_size = ws  # Choose an odd number for the window size
    polynomial_order = 4  # Choose the order of the polynomial to fit

    eel_i_denoised = savgol_filter(eel_i, window_size, polynomial_order)

    return eel_i_denoised


def interpolate_spectrum(ener,eel_i,emax=42,E0=2.6):

    Ne = int(emax*50)
    egrid = np.linspace(0, emax, Ne)

    f = interp1d(ener, eel_i, kind='cubic', bounds_error=False, fill_value=0)
    eel_i_interp = f(egrid)
    eel_i_interp =  eel_i_interp*(egrid > E0) # hard cutoff for small energies

    return egrid, eel_i_interp


def interpolate_dispersion(ener,eel,q_grid,dq):

    f_q = interp1d(q_grid, eel, axis=0, kind= 'slinear',fill_value="extrapolate")
    q_dense = np.linspace(q_grid[0],q_grid[-1], int((q_grid[-1]-q_grid[0])/dq)+1)
    #q_dense = np.linspace(0,q_grid[-1], int(q_grid[-1]/dq)+1)
    eel_interp = f_q(q_dense)

    return q_dense, eel_interp

# Settings ------------------------------------------