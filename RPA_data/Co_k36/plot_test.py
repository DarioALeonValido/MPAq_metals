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


# General variables ----------------------------------------------

qGH=np.array([145,359,539,689,811,909,987,1047,1093,1130,1184,1217,1227,1234,1240]) #bcc
qGX=np.array([55,88,119,148,175,200,223,244,263,280,295,308,319,328,335,343]) #fcc
qhGM=np.array([14,27,40,53,66,79,92,105,118,131,144,157,170,183,196,209,222,235]) #hpc
qhGK=np.array([248,469,677,859,1028,1171,1301,1405,1496,1561,1613,1639]) #hpc
qhAH=np.array([260,481,689,871,1040,1183,1313,1417,1508,1573,1625,1651]) #hpc

qmaxGN = 19 #including q=0
dqGN = 1/2/qmaxGN
q_GN = np.linspace(0, 0.5, qmaxGN)

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

f=0.75
fx=1;fy=2
fig1, axs1 = plt.subplots(fy,fx,sharey='all',layout='constrained')
fig1.set_size_inches(f*3*fx, f*6*fy)

for q in [1,2,14,27,40,53]:
  energy,eel_i,eel_r = np.genfromtxt(out_ha_q+eel+str(q)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
  axs1[0].plot(energy,eel_r) 
  axs1[1].plot(energy,eel_i)  

plt.show()



