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

cmap_blue = matplotlib.cm.get_cmap('Blues')
cmap_orange = matplotlib.cm.get_cmap('Oranges')
cfa=0.8

######### Color maps ###################
den_map='Oranges'#'Blues'#'twilight_shifted'#'binary'#'twilight'#plasma, 'magma' #'gray' #plt.cm.jet
nor="log" #'linear'
########################################

f=0.9
fx=2;fy=2
fig1, axs1 = plt.subplots(1,2)
fig1.set_size_inches(f*5*fx, f*3*fy)


"""
# linear interpolation of the q -> 0 limit
energy,eps_i2,eps_r2 = np.genfromtxt(out_ip_q+eps+str(2)+theoIP,autostrip=True,usecols=[0,1,2],unpack=True)
energy,eps_i3,eps_r3 = np.genfromtxt(out_ip_q+eps+str(3)+theoIP,autostrip=True,usecols=[0,1,2],unpack=True)
eps_r = 2*eps_r2 - eps_r3
eps_i = 2*eps_i2 - eps_i3
#
# linear interpolation of the q -> 0 limit
energy,eps_i2,eps_r2 = np.genfromtxt(out_ha_q+eps+str(2)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
energy,eps_i3,eps_r3 = np.genfromtxt(out_ha_q+eps+str(3)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
eps_r = 2*eps_r2 - eps_r3
eps_i = 2*eps_i2 - eps_i3
"""

################## RPA ############################################
qmax = 19
dq = 1/2/qmax
q_GA = np.linspace(0, 0.5, qmax)
print(q_GA)

map_eps_i = []
map_eel_i = []

for q in range(2,qmax+1):
    energy,eps_i,eps_r = np.genfromtxt(out_ha_q+eps+str(q)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
    map_eps_i.append( eps_i)
    #
    energy,eel_i,eel_r = np.genfromtxt(out_ha_q+eel+str(q)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
    map_eel_i.append(eel_i)
    #

wind = 800
map_eps_i=np.array(map_eps_i)
map_eel_i=np.array(map_eel_i)
map_eps_i=map_eps_i.T;map_eel_i=map_eel_i.T
grid_q, grid_w = np.meshgrid(q_GA[1:], energy[:wind])
Z=map_eel_i[:wind,:]
    
pcm = axs1[1].pcolor(grid_q, grid_w, Z,
                   #norm=colors.LogNorm(vmin=Z.min()+0.0001, vmax=Z.max()),
                   cmap=den_map)
fig1.colorbar(pcm, ax=axs1[1], extend='max')

axs1[0].contour(grid_q, grid_w, Z, 30, colors='black', linewidths=0.75);
#axs1[1].contour(grid_q, grid_w, Z, 30, colors='black', linewidths=0.75);


fig1.savefig('cmap.png') 
fig1.savefig('cmap.pdf')



###########################################################################################
###########################################################################################
###########################################################################################
plt.show()


