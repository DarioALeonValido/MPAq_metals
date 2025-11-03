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

# General variables ----------------------------------------------

Qpnts=[1]
HatoeV=27.2113845

V='V'
Cr='Cr'
Cu='Cu'
Zn='Zn'

Xmp_head="ndb.Xmpa_"
ER_head="ndb.mpa_ER_"
Xff_head="ndb.em1d_"
rim_head="ndb.RIM"

calc="X-MP_b500_x5Ry_r0-300i30d0.001-1eV_ho3000/"

# Functions ------------------------------------------------------

def ReadRIM(rim_file,q):
    polfile = Dataset(rim_file,"r") 
    rim = polfile.variables["RIM_qpg"][0,0,q-1]

    return rim/2


def ReadPolM(path,Xmpa_head,frag):
    polfile = Dataset(path+Xmpa_head+"fragment_"+str(frag),"r") 
    wr = polfile.variables["FREQ_sec_iq"+str(frag)][:,0]
    wi = polfile.variables["FREQ_sec_iq"+str(frag)][:,1]
    Xr = polfile.variables["X_Q_"+str(frag)][:,:,:,0]
    Xi = polfile.variables["X_Q_"+str(frag)][:,:,:,1]
    dimX = len(Xr[0])

    return wr,wi,dimX,Xr,Xi


def sampling(ene,wgrid):

    indexes  = zeros_like(wgrid,dtype='int')
    for j in range(len(wgrid)):
        i=0
        while(ene[i] < wgrid[j].real and i < len(ene)):
            i=i+1
        indexes[j]  = i 
 
    return indexes

def KK_piecewise_linear_re(wr,xi,damp):

    reX=np.array(wr);imX=np.array(wr);#reEpsilon=[];imEpsilon=[];
    for j in range(len(wr)):
      w=complex(wr[j],damp)
      s=0
      for i in range(1,len(wr)):
        a=(xi[i]-xi[i-1])/(wr[i]-wr[i-1])
        b=xi[i-1]-a*wr[i-1]

        s=s+a*(wr[i]-wr[i-1])-a*w*(np.arctanh(wr[i]/w)-np.arctanh(wr[i-1]/w))+0.5*b*np.log((wr[i]**2-w**2)/(wr[i-1]**2-w**2))
    
      s=s*2/3.14159265
      reX[j] = np.real(s)
      imX[j] = np.imag(s)
      #reEpsilon.append(-s/(s**2+xi[j]**2))
      #imEpsilon.append(xi[j]/(s**2+xi[j]**2))

    return reX,imX#reEpsilon,imEpsilon


def reX(wr,wi,npols,Er,Ei,Rr,Ri):
    reX = 0.
    for i in range(npols):
        reX = reX -(2*(Ei[i]**3*Ri[i] +Ei[i]**2*Er[i]*Rr[i] +Ei[i]*(Er[i]**2*Ri[i] 
                   -Ri[i]*wi**2 -2*Rr[i]*wi*wr +Ri[i]*wr**2) +Er[i]*(Er[i]**2*Rr[i]
                   +Rr[i]*wi**2 -2*Ri[i]*wi*wr -Rr[i]*wr**2)))/(Ei[i]**4 + Er[i]**4 
                   -8*Ei[i]*Er[i]*wi*wr +2*Er[i]**2*(wi**2 -wr**2) +2*Ei[i]**2*
                   (Er[i]**2 -wi**2 + wr**2) +(wi**2 + wr**2)**2)

    return reX


def imX(wr,wi,npols,Er,Ei,Rr,Ri):
    imX = 0.
    for i in range(npols):
        imX = imX +(2*(Ei[i]**3*Rr[i] -Ei[i]**2*Er[i]*Ri[i] -Er[i]*(Er[i]**2*Ri[i]
                   +Ri[i]*wi**2 +2*Rr[i]*wi*wr -Ri[i]*wr**2) +Ei[i]*(Er[i]**2*Rr[i]
                   -Rr[i]*wi**2 + 2*Ri[i]*wi*wr +Rr[i]*wr**2)))/(Ei[i]**4 +Er[i]**4 
                   -8*Ei[i]*Er[i]*wi*wr +2*Er[i]**2*(wi**2 -wr**2) +2*Ei[i]**2*
                   (Er[i]**2 -wi**2 +wr**2) +(wi**2 +wr**2)**2)

    return imX


def reimXmp(wc,E,R):

    Xmp = np.zeros_like(wc,dtype='complex128')
    for i in range(len(wc)):
        for p in range(len(E)):
            Xmp[i] = Xmp[i] + 2*E[p]*R[p]/(wc[i]**2-E[p]**2)

    return Xmp


################ module fit ################################################

def Xmp(w,E,R):

    if(len(E)==len(R)):
        return np.sum(2*E*R/(w**2-E**2))

# Reading and processing the data ------------------------------------------

i=0
j=0

plt.rcParams['font.size'] = '11.5'
plt.rcParams['lines.linewidth'] = 1.5

f=0.9
fx=1;fy=2
fig1, axs1 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig1.set_size_inches(f*4*fx, f*3*fy)

fig2, axs2 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig2.set_size_inches(f*4*fx, f*3*fy)

fig3, axs3 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig2.set_size_inches(f*4*fx, f*3*fy)

"""
fig4, axs4 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig2.set_size_inches(f*4*fx, f*3*fy)
"""

colors = ['royalblue','tab:orange', 'forestgreen', 'k','r']
colG = ['tomato','lime','limegreen','forestgreen','darkgreen']


for q in [1]:
    #v_rim=ReadRIM(rim_head,q)
    #print('rim=',v_rim)
    v_rim=1
    wrDP,wiDP,dimXDP,XrDP,XiDP = ReadPolM(calc,Xmp_head,q)
    nfreq=int(len(wrDP)/2)
    energy=wrDP[:nfreq]
    XrS1=XrDP[:nfreq,i,j]; XiS1=XiDP[:nfreq,i,j]
    XrS2=XrDP[nfreq:,i,j]; XiS2=XiDP[nfreq:,i,j]
    axs1[0].plot(energy, XrS1)
    axs1[1].plot(energy,-XiS1)
    axs1[0].plot(energy, XrS2)
    axs1[1].plot(energy,-XiS2)

    axs2[0].plot(energy, XrS1)
    axs2[1].plot(energy,-XiS1)


    #MPA sampling
    shift=wiDP[0]
    damping=wiDP[1]
    imag2l=wiDP[-1]
    print('samp',shift,damping,imag2l)

    npol=15
    w0=[1j*imag2l, energy[-1]+1j*imag2l]
    wgrid = samp.mpa_frequency_sampling(npol, w0, [shift,damping], ps='2l', alpha=1.5)

    windexes = sampling(energy,wgrid[:npol])
    print('samp indexes:',windexes)
    wr = np.concatenate((wrDP[windexes],wrDP[nfreq+windexes]),dtype='complex128')
    wi = np.concatenate((wiDP[windexes],wiDP[nfreq+windexes]),dtype='complex128')
    wc = wr + 1j*wi
    Xc = np.concatenate((XrS1[windexes]+1j*XiS1[windexes],XrS2[windexes]+1j*XiS2[windexes]),dtype='complex128')

    #print(wc[:npol],Xc[:npol])
    axs1[0].scatter(np.real(wc[:npol]), np.real(Xc[:npol]))
    axs1[1].scatter(np.real(wc[:npol]),-np.imag(Xc[:npol]))
    axs1[0].scatter(np.real(wc[:npol]), np.real(Xc[npol:]))
    axs1[1].scatter(np.real(wc[:npol]),-np.imag(Xc[npol:]))

    #MPA interpolation
    R, E, MPred, PPcond_rate = mpa.mpa_RE_solver(npol, wc, Xc)
    print('Sampling:','\n',wc)
    print('Poles:','\n',E.real,'\n',E.imag)


    XreMP = reX(energy,imag2l,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
    XimMP = imX(energy,imag2l,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
    #XreMP = reX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
    #XimMP = imX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
    #axs1[0].plot(energy, XreMP,label='MP'+'_q'+str(q),ls='--')
    #axs1[1].plot(energy,-XimMP,label='MP'+'_q'+str(q),ls='--')

    Wmp=energy #+ 1j*imag2l
    Xmp=reimXmp(Wmp,E,R)
    axs1[0].plot(energy, np.real(Xmp),label='MP'+'_q'+str(q),ls='--')
    axs1[1].plot(energy,-np.imag(Xmp),label='MP'+'_q'+str(q),ls='--')


q=2
for npol in [3]:
    #v_rim=ReadRIM(rim_head,q)
    #print('rim=',v_rim)
    v_rim=1
    wrDP,wiDP,dimXDP,XrDP,XiDP = ReadPolM(calc,Xmp_head,q)
    nfreq=int(len(wrDP)/2)
    energy=wrDP[:nfreq]
    XrS1=XrDP[:nfreq,i,j]; XiS1=XiDP[:nfreq,i,j]
    XrS2=XrDP[nfreq:,i,j]; XiS2=XiDP[nfreq:,i,j]

    #MPA sampling
    shift=wiDP[0]
    damping=wiDP[1]
    imag2l=wiDP[-1]
    print('samp',shift,damping,imag2l)

    w0=[1j*imag2l, energy[-1]*0.99+1j*imag2l]
    wgrid = samp.mpa_frequency_sampling(npol, w0, [shift,damping], ps='2l', alpha=2)

    windexes = sampling(energy,wgrid[:npol])
    wr = np.concatenate((wrDP[windexes],wrDP[nfreq+windexes]),dtype='complex128')
    wi = np.concatenate((wiDP[windexes],wiDP[nfreq+windexes]),dtype='complex128')
    wc = wr + 1j*wi
    Xc = np.concatenate((XrS1[windexes]+1j*XiS1[windexes],XrS2[windexes]+1j*XiS2[windexes]),dtype='complex128')

    #MPA interpolation
    R, E, MPred, PPcond_rate = mpa.mpa_RE_solver(npol, wc, Xc)

    XreMP = reX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
    XimMP = imX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
    #axs2[0].plot(energy, XreMP,label='MP'+str(npol))
    #axs2[1].plot(energy,-XimMP,label='MP'+str(npol))


for npol in range(5,15,2):
    #MPA sampling from previous poles
    new_grid = np.zeros(npol,dtype='complex128')
    new_grid[ 0] = wgrid[ 0]
    new_grid[-1] = wgrid[-1]; new_grid[1:-1] = E.real
    #new_grid[1:] = E.real
    print('new grid:',new_grid.real)
    
    windexes = sampling(energy,new_grid)
    wr = np.concatenate((wrDP[windexes],wrDP[nfreq+windexes]),dtype='complex128')
    wi = np.concatenate((wiDP[windexes],wiDP[nfreq+windexes]),dtype='complex128')
    wc = wr + 1j*wi
    Xc = np.concatenate((XrS1[windexes]+1j*XiS1[windexes],XrS2[windexes]+1j*XiS2[windexes]),dtype='complex128')

    #MPA interpolation
    R, E, MPred, PPcond_rate = mpa.mpa_RE_solver(npol, wc, Xc)

    #XreMP = reX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
    #XimMP = imX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))

    # repetition
    for rep in range(5):
        new_grid = np.sort(E.real)
        new_grid[-2] = wgrid[ 0]
        new_grid[-1] = wgrid[-1]

        windexes = sampling(energy,new_grid)
        wr = np.concatenate((wrDP[windexes],wrDP[nfreq+windexes]),dtype='complex128')
        wi = np.concatenate((wiDP[windexes],wiDP[nfreq+windexes]),dtype='complex128')
        wc = wr + 1j*wi
        Xc = np.concatenate((XrS1[windexes]+1j*XiS1[windexes],XrS2[windexes]+1j*XiS2[windexes]),dtype='complex128')

        #MPA interpolation
        R, E, MPred, PPcond_rate = mpa.mpa_RE_solver(npol, wc, Xc)

        XreMP = reX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
        XimMP = imX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))


axs2[0].plot(energy, XreMP,label='MP'+str(npol))
axs2[1].plot(energy,-XimMP,label='MP'+str(npol))


for q in [1,2,3,4,5,6,7,8]:
    wrDP,wiDP,dimXDP,XrDP,XiDP = ReadPolM(calc,Xmp_head,q)
    nfreq=int(len(wrDP)/2)
    energy=wrDP[:nfreq]
    XrS1=XrDP[:nfreq,i,j]; XiS1=XiDP[:nfreq,i,j]
    XrS2=XrDP[nfreq:,i,j]; XiS2=XiDP[nfreq:,i,j]
    #windexes from previous poles for q=1
    wr = np.concatenate((wrDP[windexes],wrDP[nfreq+windexes]),dtype='complex128')
    wi = np.concatenate((wiDP[windexes],wiDP[nfreq+windexes]),dtype='complex128')
    wc = wr + 1j*wi
    Xc = np.concatenate((XrS1[windexes]+1j*XiS1[windexes],XrS2[windexes]+1j*XiS2[windexes]),dtype='complex128')
    #MPA interpolation
    R, E, MPred, PPcond_rate = mpa.mpa_RE_solver(npol, wc, Xc)

    XreMP = reX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
    XimMP = imX(energy,0.,npol,np.real(E),np.imag(E),np.real(R),np.imag(R))
    axs3[0].plot(energy, XreMP,label='q'+str(q))
    axs3[1].plot(energy,-XimMP,label='q'+str(q))
   
#axs1[1,1].set_ylim(-0.0,0.32)
#axs1[1,1].set_xlim(0.0,46)


axs1[0].legend(edgecolor='k') #loc=4,borderaxespad=0.3,title=EF_q[ef]
#axs1[0,1].legend(edgecolor='k')
axs1[1].set_xlabel("$\omega~(eV)$")

axs1[0].set_ylabel("$Re[-\epsilon^{-1}]~(a.u)$")
axs1[1].set_ylabel("$Im[-\epsilon^{-1}]~(a.u)$")
"""
axs2[0,0].legend(edgecolor='k') #loc=4,borderaxespad=0.3,title=EF_q[ef]
#axs1[0,1].legend(edgecolor='k')
axs2[1,0].set_xlabel("$\omega~(eV)$")
axs2[1,1].set_xlabel("$\omega~(eV)$")

axs2[0,0].set_ylabel("$Re[\epsilon]~(a.u)$")
axs2[1,0].set_ylabel("$Im[\epsilon]~(a.u)$")
axs2[0,1].set_ylabel("$Re[-\epsilon^{-1}]~(a.u)$")
axs2[1,1].set_ylabel("$Im[-\epsilon^{-1}]~(a.u)$")

axs3[0,0].legend(edgecolor='k') #loc=4,borderaxespad=0.3,title=EF_q[ef]
#axs1[0,1].legend(edgecolor='k')
axs3[1,0].set_xlabel("$\omega~(eV)$")
axs3[1,1].set_xlabel("$\omega~(eV)$")

axs3[0,0].set_ylabel("$Re[\epsilon]~(a.u)$")
axs3[1,0].set_ylabel("$Im[\epsilon]~(a.u)$")
axs3[0,1].set_ylabel("$Re[-\epsilon^{-1}]~(a.u)$")
axs3[1,1].set_ylabel("$Im[-\epsilon^{-1}]~(a.u)$")

axs4[0,0].legend(edgecolor='k') #loc=4,borderaxespad=0.3,title=EF_q[ef]
axs4[1,1].legend(edgecolor='k')
axs4[1,0].set_xlabel("$\omega~(eV)$")
axs4[1,1].set_xlabel("$\omega~(eV)$")

axs4[0,0].set_ylabel("$Re[\epsilon]~(a.u)$")
axs4[1,0].set_ylabel("$Im[\epsilon]~(a.u)$")
axs4[0,1].set_ylabel("$Re[-\epsilon^{-1}]~(a.u)$")
axs4[1,1].set_ylabel("$Im[-\epsilon^{-1}]~(a.u)$")

fig1.savefig('Xq_IPh_f1.png') 
fig1.savefig('Xq_IPh_f1.pdf')

fig2.savefig('Xq_IPh_f2.png') 
fig2.savefig('Xq_IPh_f2.pdf')

fig2.savefig('Xq_IPh_f3.png') 
fig2.savefig('Xq_IPh_f3.pdf')
"""

###########################################################################################
###########################################################################################
###########################################################################################
plt.show()


