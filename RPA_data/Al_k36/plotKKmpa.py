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

V='V_k36'
Cr='Cr'
Cu='Cu'
Zn='Zn'

Xmp_head="ndb.Xmpa_"
ER_head="ndb.mpa_ER_"
Xff_head="ndb.em1d_"
rim_head="ndb.RIM"

out_ip_q='o-opt-IP_b500_x1RL_r0-200eV_d0.1eV_f2000.'
out_ha_q='o-opt-HA_b500_x5Ry_r0-200eV_d0.1eV_f2000.'

eps='eps_q'
eel='eel_q'
theoIP='_ip'
theoHA='_inv_rpa_dyson'

# Functions ------------------------------------------------------

def ReadRIM(rim_file,q):
    polfile = Dataset(rim_file,"r") 
    rim = polfile.variables["RIM_qpg"][0,0,q-1]

    return rim/2

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
    
      s=s*2/np.pi
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


# Reading and processing the data ------------------------------------------

i=0
j=0

plt.rcParams['font.size'] = '11.5'
plt.rcParams['lines.linewidth'] = 1.5

cmap_blue = matplotlib.colormaps.get_cmap('Blues')
cmap_orange = matplotlib.colormaps.get_cmap('Oranges')
cfa=0.8

f=0.9
fx=2;fy=2
"""
fig1, axs1 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig1.set_size_inches(f*4*fx, f*3*fy)

fig2, axs2 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig2.set_size_inches(f*4*fx, f*3*fy)

fig3, axs3 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig2.set_size_inches(f*4*fx, f*3*fy)

fig4, axs4 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig4.set_size_inches(f*4*fx, f*3*fy)
"""

fig5, axs5 = plt.subplots(fy, fx,sharex='col',constrained_layout=True)
fig5.set_size_inches(f*4*fx, f*3*fy)

colors = ['royalblue','tab:orange', 'forestgreen', 'k','r']
colG = ['tomato','lime','limegreen','forestgreen','darkgreen']


#axs1[0,0].plot([-0.5,100.5],[0,0],c='k',linewidth=1.0)
#axs2[0,0].plot([-0.5,100.5],[0,0],c='k',linewidth=1.0)

q=1

GTm2=1.3756e-6 #eV
GTm2=8.45277e-12 #eV

"""
# linear interpolation of the q -> 0 limit
energy,eps_i2,eps_r2 = np.genfromtxt(out_ip_q+eps+str(2)+theoIP,autostrip=True,usecols=[0,1,2],unpack=True)
energy,eps_i3,eps_r3 = np.genfromtxt(out_ip_q+eps+str(3)+theoIP,autostrip=True,usecols=[0,1,2],unpack=True)
eps_r = 2*eps_r2 - eps_r3
eps_i = 2*eps_i2 - eps_i3
#
v_rim=ReadRIM(rim_head,1)
print('rim=',v_rim,1/v_rim**2)
#
axs1[0,0].plot(energy,eps_r,label='IP d0.1eV q'+str(q),c=cmap_blue(19/19))
axs1[1,0].plot(energy,eps_i,c=cmap_blue(19/19))
#
axs2[0,0].plot(energy,eps_r/v_rim,label='IP d0.1eV q'+str(q),c=cmap_blue(19/19))
axs2[1,0].plot(energy,eps_i/v_rim,c=cmap_blue(19/19))
#
energy,eel_i2,eel_r2 = np.genfromtxt(out_ip_q+eel+str(2)+theoIP,autostrip=True,usecols=[0,1,2],unpack=True)
energy,eel_i3,eel_r3 = np.genfromtxt(out_ip_q+eel+str(3)+theoIP,autostrip=True,usecols=[0,1,2],unpack=True)
eel_r = 2*eel_r2 - eel_r3
eel_i = 2*eel_i2 - eel_i3
axs1[0,1].plot(energy,-eel_r,label='IP d0.1eV q'+str(q),c=cmap_blue(19/19))
axs1[1,1].plot(energy, eel_i,c=cmap_blue(19/19))
#
#W_r = eel_r/(2*np.pi/v_rim + GTm2*energy**2)
#W_i = eel_i/(2*np.pi/v_rim + GTm2*energy**2)
qmax=19
dq=1/2/qmax
W_r = eel_r/(dq**2 + GTm2*energy**2)
W_i = eel_i/(dq**2 + GTm2*energy**2)
#axs2[0,1].plot(energy,W_r,label='IP d0.1eV q'+str(q),c=cmap_blue(19/19))
#axs2[1,1].plot(energy,W_i,c=cmap_blue(19/19))

# integration in omega
dwE=0.2 #eV experiments
dwC=0.1 #eV calculations
W_rI=(W_r[:-2]+2*W_r[1:-1]+W_r[2:])*dwC
W_iI=(W_i[:-2]+2*W_i[1:-1]+W_i[2:])*dwC
axs2[0,1].plot(energy[:-2],-W_rI,label='IP d0.1eV q'+str(q),c=cmap_blue(19/19))
axs2[1,1].plot(energy[:-2], W_iI,c=cmap_blue(19/19))


for q in range(2,qmax+1):
    v_rim=ReadRIM(rim_head,q)
    print('rim=',v_rim,1/v_rim**2)
    energy,eps_i,eps_r = np.genfromtxt(out_ip_q+eps+str(q)+theoIP,autostrip=True,usecols=[0,1,2],unpack=True)
    axs1[0,0].plot(energy,eps_r,label='IP d0.1eV q'+str(q),c=cmap_blue((qmax-cfa*(q-1))/qmax))
    axs1[1,0].plot(energy,eps_i,c=cmap_blue((qmax-cfa*(q-1))/qmax))
    #
    axs2[0,0].plot(energy,eps_r/v_rim,label='IP d0.1eV q'+str(q),c=cmap_blue((qmax-cfa*(q-1))/qmax))
    axs2[1,0].plot(energy,eps_i/v_rim,c=cmap_blue((qmax-cfa*(q-1))/qmax))
    
    energy,eel_i,eel_r = np.genfromtxt(out_ip_q+eel+str(q)+theoIP,autostrip=True,usecols=[0,1,2],unpack=True)
    axs1[0,1].plot(energy,-eel_r,label='IP d0.1eV q'+str(q),c=cmap_blue((qmax-cfa*(q-1))/qmax))
    axs1[1,1].plot(energy, eel_i,c=cmap_blue((qmax-cfa*(q-1))/qmax))
    #
    #W_r = eel_r/(2*np.pi/v_rim + GTm2*energy**2)
    #W_i = eel_i/(2*np.pi/v_rim + GTm2*energy**2)
    #
    # omega integration
    W_r = eel_r/((q*dq)**2 + GTm2*energy**2)
    W_i = eel_i/((q*dq)**2 + GTm2*energy**2)
    W_rI=(W_r[:-2]+2*W_r[1:-1]+W_r[2:])*dwC
    W_iI=(W_i[:-2]+2*W_i[1:-1]+W_i[2:])*dwC
    axs2[0,1].plot(energy[:-2],-W_rI,label='IP d0.1eV q'+str(q),c=cmap_blue((qmax-cfa*(q-1))/qmax))
    axs2[1,1].plot(energy[:-2], W_iI,c=cmap_blue((qmax-cfa*(q-1))/qmax))
"""

"""
################## RPA ############################################

# linear interpolation of the q -> 0 limit
energy,eps_i2,eps_r2 = np.genfromtxt(out_ha_q+eps+str(2)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
energy,eps_i3,eps_r3 = np.genfromtxt(out_ha_q+eps+str(3)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
eps_r = 2*eps_r2 - eps_r3
eps_i = 2*eps_i2 - eps_i3
#
v_rim=ReadRIM(rim_head,1)
print('rim=',v_rim,1/v_rim**2)
#
axs3[0,0].plot(energy,eps_r,label='HA d0.1eV q'+str(q),c=cmap_blue(19/19))
axs3[1,0].plot(energy,eps_i,c=cmap_blue(19/19))
#
axs4[0,0].plot(energy,eps_r/v_rim,label='HA d0.1eV q'+str(q),c=cmap_blue(19/19))
axs4[1,0].plot(energy,eps_i/v_rim,c=cmap_blue(19/19))
#
energy,eel_i2,eel_r2 = np.genfromtxt(out_ha_q+eel+str(2)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
energy,eel_i3,eel_r3 = np.genfromtxt(out_ha_q+eel+str(3)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
eel_r = 2*eel_r2 - eel_r3
eel_i = 2*eel_i2 - eel_i3
axs3[0,1].plot(energy,-eel_r,label='HA d0.1eV q'+str(q),c=cmap_blue(19/19))
axs3[1,1].plot(energy, eel_i,c=cmap_blue(19/19))
#axs3[0,0].plot(energy,eel_r/(eel_r*2+eel_i*2),label='HA d0.1eV q'+str(q),c=cmap_blue(19/19))
#axs3[1,0].plot(energy,eel_i/(eel_r*2+eel_i*2),c=cmap_blue(19/19))
#
#W_r = eel_r/(2*np.pi/v_rim + GTm2*energy**2)
#W_i = eel_i/(2*np.pi/v_rim + GTm2*energy**2)
qmax=19
dq=1/2/qmax
W_r = eel_r/(dq**2 + GTm2*energy**2)
W_i = eel_i/(dq**2 + GTm2*energy**2)
#axs4[0,1].plot(energy,W_r,label='IP d0.1eV q'+str(q),c=cmap_blue(19/19))
#axs4[1,1].plot(energy,W_i,c=cmap_blue(19/19))

# integration in omega
dwE=0.2 #eV experiments
dwC=0.1 #eV calculations
W_rI=(W_r[:-2]+2*W_r[1:-1]+W_r[2:])*dwC
W_iI=(W_i[:-2]+2*W_i[1:-1]+W_i[2:])*dwC
axs4[0,1].plot(energy[:-2],-W_rI,label='HA d0.1eV q'+str(q),c=cmap_blue(19/19))
axs4[1,1].plot(energy[:-2], W_iI,c=cmap_blue(19/19))


for q in range(2,qmax+1):
    v_rim=ReadRIM(rim_head,q)
    print('rim=',v_rim,1/v_rim**2)
    energy,eps_i,eps_r = np.genfromtxt(out_ha_q+eps+str(q)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
    axs3[0,0].plot(energy,eps_r,label='HA d0.1eV q'+str(q),c=cmap_blue((qmax-cfa*(q-1))/qmax))
    axs3[1,0].plot(energy,eps_i,c=cmap_blue((qmax-cfa*(q-1))/qmax))
    #
    axs4[0,0].plot(energy,eps_r/v_rim,label='HA d0.1eV q'+str(q),c=cmap_blue((qmax-cfa*(q-1))/qmax))
    axs4[1,0].plot(energy,eps_i/v_rim,c=cmap_blue((qmax-cfa*(q-1))/qmax))
    
    energy,eel_i,eel_r = np.genfromtxt(out_ha_q+eel+str(q)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
    axs3[0,1].plot(energy,-eel_r,label='HA d0.1eV q'+str(q),c=cmap_blue((qmax-cfa*(q-1))/qmax))
    axs3[1,1].plot(energy, eel_i,c=cmap_blue((qmax-cfa*(q-1))/qmax))
    #
    #W_r = eel_r/(2*np.pi/v_rim + GTm2*energy**2)
    #W_i = eel_i/(2*np.pi/v_rim + GTm2*energy**2)
    #
    # omega integration
    W_r = eel_r/((q*dq)**2 + GTm2*energy**2)
    W_i = eel_i/((q*dq)**2 + GTm2*energy**2)
    W_rI=(W_r[:-2]+2*W_r[1:-1]+W_r[2:])*dwC
    W_iI=(W_i[:-2]+2*W_i[1:-1]+W_i[2:])*dwC
    axs4[0,1].plot(energy[:-2],-W_rI,label='HA d0.1eV q'+str(q),c=cmap_blue((qmax-cfa*(q-1))/qmax))
    axs4[1,1].plot(energy[:-2], W_iI,c=cmap_blue((qmax-cfa*(q-1))/qmax))
"""

dwE=0.2 #eV experiments
dwC=0.1 #eV calculations

qmax=19
dq=1/2/qmax

shift=0.1
damping = 0.5
imag2l = 30
wmax = 190 #energy[-1]*0.8
npol=15
alph=2.2
for q in [1,2]:
    energy,eel_i,eel_r = np.genfromtxt(out_ha_q+eel+str(q)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)

    ex_r1, ex_i1 = KK_piecewise_linear_re(np.array(energy,dtype='float128'),np.array(-eel_i,dtype='float128'),damping)
    ex_r2, ex_i2 = KK_piecewise_linear_re(np.array(energy,dtype='float128'),np.array(-eel_i,dtype='float128'),imag2l)
    #ex_r1 = -ex_r1; ex_r2 = -ex_r2

    axs5[0,1].plot(energy,-eel_r)
    axs5[1,1].plot(energy, eel_i)
    axs5[0,1].plot(energy, ex_r1 + 1,ls=':')
    axs5[1,1].plot(energy,-ex_i1,ls=':')
    axs5[0,1].plot(energy, ex_r2 + 1,ls=':')
    axs5[1,1].plot(energy,-ex_i2,ls=':')
    #
    W_r = ex_r1/((q*dq)**2 + GTm2*energy**2)
    W_i = ex_i1/((q*dq)**2 + GTm2*energy**2)
    W_rI=(W_r[:-2]+2*W_r[1:-1]+W_r[2:])*dwC
    W_iI=(W_i[:-2]+2*W_i[1:-1]+W_i[2:])*dwC
    #axs4[0,1].plot(energy[:-2], W_rI + 1,c='r')
    #axs4[1,1].plot(energy[:-2],-W_iI,c='r')

    #MPA sampling
    w0=[1j*imag2l, wmax +1j*imag2l]
    wgrid = samp.mpa_frequency_sampling(npol, w0, [shift,damping], ps='2l', alpha=alph)
    windexes = sampling(energy,wgrid[:npol])
    wr = np.concatenate((energy[windexes],energy[windexes]),dtype='complex128')
    wi = np.concatenate((np.array([shift]),np.full((npol-1),damping),np.full((npol),imag2l)),dtype='complex128')
    wc = wr + 1j*wi
    Xc = np.concatenate((ex_r1[windexes]+1j*ex_i1[windexes],ex_r2[windexes]+1j*ex_i2[windexes]),dtype='complex128')

    # MPA interpolation
    R, E, MPred, PPcond_rate = mpa.mpa_RE_solver(npol, wc/HatoeV, Xc)

    XreMP = reX(energy,0.,npol,np.real(E)*HatoeV,np.imag(E)*HatoeV,np.real(R)*HatoeV,np.imag(R)*HatoeV)
    XimMP = imX(energy,0.,npol,np.real(E)*HatoeV,np.imag(E)*HatoeV,np.real(R)*HatoeV,np.imag(R)*HatoeV)
    axs5[0,1].plot(energy, XreMP + 1,label='MP'+str(npol)+'_alpha='+str(alph))
    axs5[1,1].plot(energy,-XimMP,label='MP'+str(npol)+'_alpha='+str(alph))
    print('q:',q)
    print('R:',R)
    print('E:',E)


    
#axs1[1,1].set_ylim(-0.0,0.32)
#axs1[1,1].set_xlim(0.0,46)

"""
axs1[0,0].legend(edgecolor='k') #loc=4,borderaxespad=0.3,title=EF_q[ef]
#axs1[0,1].legend(edgecolor='k')
axs1[1,0].set_xlabel("$\omega~(eV)$")
axs1[1,1].set_xlabel("$\omega~(eV)$")

axs1[0,0].set_ylabel("$Re[\epsilon]~(a.u)$")
axs1[1,0].set_ylabel("$Im[\epsilon]~(a.u)$")
axs1[0,1].set_ylabel("$Re[\epsilon^{-1}]~(a.u)$")
axs1[1,1].set_ylabel("$-Im[\epsilon^{-1}]~(a.u)$")

axs2[0,0].legend(edgecolor='k') #loc=4,borderaxespad=0.3,title=EF_q[ef]
#axs1[0,1].legend(edgecolor='k')
axs2[1,0].set_xlabel("$\omega~(eV)$")
axs2[1,1].set_xlabel("$\omega~(eV)$")

axs2[0,0].set_ylabel("$Re[\epsilon]~(a.u)$")
axs2[1,0].set_ylabel("$Im[\epsilon]~(a.u)$")
axs2[0,1].set_ylabel("$Re[\epsilon^{-1}]~(a.u)$")
axs2[1,1].set_ylabel("$-Im[\epsilon^{-1}]~(a.u)$")

axs3[0,0].legend(edgecolor='k') #loc=4,borderaxespad=0.3,title=EF_q[ef]
#axs1[0,1].legend(edgecolor='k')
axs3[1,0].set_xlabel("$\omega~(eV)$")
axs3[1,1].set_xlabel("$\omega~(eV)$")

axs3[0,0].set_ylabel("$Re[\epsilon]~(a.u)$")
axs3[1,0].set_ylabel("$Im[\epsilon]~(a.u)$")
axs3[0,1].set_ylabel("$Re[\epsilon^{-1}]~(a.u)$")
axs3[1,1].set_ylabel("$-Im[\epsilon^{-1}]~(a.u)$")

axs4[0,0].legend(edgecolor='k') #loc=4,borderaxespad=0.3,title=EF_q[ef]
#axs1[0,1].legend(edgecolor='k')
axs4[1,0].set_xlabel("$\omega~(eV)$")
axs4[1,1].set_xlabel("$\omega~(eV)$")

axs4[0,0].set_ylabel("$Re[\epsilon]~(a.u)$")
axs4[1,0].set_ylabel("$Im[\epsilon]~(a.u)$")
axs4[0,1].set_ylabel("$Re[\epsilon^{-1}]~(a.u)$")
axs4[1,1].set_ylabel("$-Im[\epsilon^{-1}]~(a.u)$")
"""

axs5[0,0].legend(edgecolor='k') #loc=4,borderaxespad=0.3,title=EF_q[ef]
#axs1[0,1].legend(edgecolor='k')
axs5[1,0].set_xlabel("$\omega~(eV)$")
axs5[1,1].set_xlabel("$\omega~(eV)$")

axs5[0,0].set_ylabel("$Re[\epsilon]~(a.u)$")
axs5[1,0].set_ylabel("$Im[\epsilon]~(a.u)$")
axs5[0,1].set_ylabel("$Re[\epsilon^{-1}]~(a.u)$")
axs5[1,1].set_ylabel("$-Im[\epsilon^{-1}]~(a.u)$")

"""
fig1.savefig('Xq_IPh_f1.png') 
fig1.savefig('Xq_IPh_f1.pdf')
"""

###########################################################################################
###########################################################################################
###########################################################################################
plt.show()


