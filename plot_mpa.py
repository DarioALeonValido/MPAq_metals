import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


# Functions ------------------------------------------------------

def mpa_Eq(q, E, c1, c2, c3):
    return E*(1. + c1*q + c2*q**2 + c3*q**3)


def mpa_model_RE(q, w, R, E, e1, e2, e3, r1, r2, r3):
    """
    Computes:
        mpa(q, w) = sum_p S[p] / (w^2 - (E[p] * (1 + e1[p]*q + e2[p]*q^2 + e3[p]*q^3))^2)
    """
    q = np.asarray(q)
    w = np.asarray(w)
    E = np.asarray(E)[:, None, None]
    e1 = np.asarray(e1)[:, None, None]
    e2 = np.asarray(e2)[:, None, None]
    e3 = np.asarray(e3)[:, None, None]
    R = np.asarray(R)[:, None, None]
    r1 = np.asarray(r1)[:, None, None]
    r2 = np.asarray(r2)[:, None, None]
    r3 = np.asarray(r3)[:, None, None]

    Q = q[None, :, None]
    W = w[None, None, :]

    E_q = E * (1. + e1*Q + e2*Q**2 + e3*Q**3) 
    R_q = R * (1. + r1*Q + r2*Q**2 + r3*Q**3)
    S_q = 2 * R_q * E_q

    denom = W**2 - E_q**2

    mpa = np.sum(S_q / denom, axis=0)  # shape: (len(q), len(w))
    return mpa

# General variables ----------------------------------------------

HatoeV=27.2113845

Li='Li_k36'
Be='Be_k36'
Na='Na_k36'
Mg='Mg_k36'
K ='K_k36'
Ca='Ca_k36'
Al='Al_k36'
Ti='Ti_k36'
V ='V_k36'
Cr='Cr_k36'
Fe='Fe_k36'
Co='Co_k36'
Ni='Ni_k36'
Cu='Cu_k36'
Zn='Zn_k36'
Mo='Mo_k36'
Pd='Pd_k36'
Ag='Ag_k36'
Sn='Sn_k36'
Sb='Sn_k36'
Te='Te_k36'
Ta='Ta_k36'
W ='W_k36'
Os='Os_k36'
Pt='Pt_k36'
Au='Au_k36'
Tl='Tl_k36'
Pb='Pb_k36'
Bi='Bi_k36'

q_dense = np.linspace(0, 0.5, 100)

# Plots ----------------------------------------------------------

plt.rcParams['font.size'] = '11.5'
plt.rcParams['lines.linewidth'] = 1.5

orange_map='BrBG'
cFont = 'w'
sFont = 15

panels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)','(q)']


f=0.75
fx=7;fy=1
fig1, axs1 = plt.subplots(fy, fx,sharey='row',sharex='all',constrained_layout=True)
fx2=7
fig1.set_size_inches(f*3*fx2*fx2/9, f*6*fy)


# Reading and processing -----------------------------------------
mat = Ca; npols = 6; emax = 14
w_dense = np.linspace(0, emax, int(emax*50), True)
grid_q, grid_w = np.meshgrid(q_dense, w_dense)

R, E, e1, e2, e3, r1, r2, r3 = np.genfromtxt(f'MPAq_data/RqEq_{mat}_unformatted.dat',dtype=complex,autostrip=True,unpack=True)
fit2D = mpa_model_RE(q_dense, w_dense, R, E, e1, e2, e3, r1, r2, r3)
Z = -np.imag(fit2D.T)

Fmin=np.min(Z)
Fmax=np.max(Z)

pcm = axs1[0].pcolor(grid_q, grid_w, Z, cmap=orange_map)
fig1.colorbar(pcm, ax=axs1,location='top', extend='max',aspect=20*fx)

axs1[0].set_ylim(w_dense[0],w_dense[-1])
axs1[0].set_xlim(q_dense[0],q_dense[-1])
axs1[0].set_xlabel("$\mathbf{q}$")
axs1[0].set_xticks([0,0.5],labels=["$\Gamma$",'$N$'])
axs1[0].set_ylabel("$\omega~$(eV)")
axs1[0].set_title(mat[:-4]+'-MPA'+' '+panels[0],weight='bold',c='k',loc='center')

for p in range(npols):
    R_q = mpa_Eq(q_dense, R[p], r1[p], r2[p], r3[p])
    E_q = mpa_Eq(q_dense, E[p], e1[p], e2[p], e3[p])
    axs1[0].plot(q_dense, np.real(E_q),c='k',ls='--')

    fit2D = mpa_model_RE(q_dense, w_dense, [R[p]], [E[p]], [e1[p]], [e2[p]], [e3[p]], [r1[p]], [r2[p]], [r3[p]])
    Z = -np.imag(fit2D.T)

    pcm = axs1[p+1].pcolor(grid_q, grid_w, Z, cmap=orange_map, vmin=Fmin, vmax=Fmax)

    axs1[p+1].set_xlabel("$\mathbf{q}$")
    axs1[p+1].set_title(f'$p ={p+1}$ '+panels[p+1],weight='bold',c='k',loc='center')


fig1.savefig(f'spectral_{mat}_mpa.png')


#######################################
mat = Ni; npols = 13; emax = 40

f=0.75
fx=7;fy=2
fig2, axs2 = plt.subplots(fy, fx,sharey='row',sharex='all',constrained_layout=True)
fig2.set_size_inches(f*3*fx*fx/9, f*6*fy)


# Reading and processing -----------------------------------------

w_dense = np.linspace(0, emax, int(emax*50), True)
grid_q, grid_w = np.meshgrid(q_dense, w_dense)

R, E, e1, e2, e3, r1, r2, r3 = np.genfromtxt(f'MPAq_data/RqEq_{mat}_unformatted.dat',dtype=complex,autostrip=True,unpack=True)
fit2D = mpa_model_RE(q_dense, w_dense, R, E, e1, e2, e3, r1, r2, r3)
Z = -np.imag(fit2D.T)

Fmin=np.min(Z)
Fmax=np.max(Z)

pcm = axs2[0,0].pcolor(grid_q, grid_w, Z, cmap=orange_map)
fig2.colorbar(pcm, ax=axs2,location='top', extend='max',aspect=20*fx)

axs2[0,0].set_ylim(w_dense[0],w_dense[-1])
axs2[0,0].set_xlim(q_dense[0],q_dense[-1])
axs2[0,0].set_xticks([0,0.5],labels=["$\Gamma$",'$N$'])
axs2[0,0].set_ylabel("$\omega~$(eV)")
axs2[0,0].set_title(mat[:-4]+'-MPA'+' '+panels[0],weight='bold',c='k',loc='center')

for p in range(npols):
    R_q = mpa_Eq(q_dense, R[p], r1[p], r2[p], r3[p])
    E_q = mpa_Eq(q_dense, E[p], e1[p], e2[p], e3[p])
    axs2[0,0].plot(q_dense, np.real(E_q),c='k',ls='--')

    fit2D = mpa_model_RE(q_dense, w_dense, [R[p]], [E[p]], [e1[p]], [e2[p]], [e3[p]], [r1[p]], [r2[p]], [r3[p]])
    Z = -np.imag(fit2D.T)

    if p < fx-1:
      pcm = axs2[0,p+1].pcolor(grid_q, grid_w, Z, cmap=orange_map, vmin=Fmin, vmax=Fmax)
      axs2[0,p+1].set_title(f'$p ={p+1}$ '+panels[p+1],weight='bold',c='k',loc='center')
    else:
      pcm = axs2[1,p-fx+1].pcolor(grid_q, grid_w, Z, cmap=orange_map, vmin=Fmin, vmax=Fmax)
      axs2[1,p-fx+1].set_xlabel("$\mathbf{q}$")
      axs2[1,p-fx+1].set_title(f'$p ={p+1}$ '+panels[p+1],weight='bold',c='k',loc='center')


fig2.savefig(f'spectral_{mat}_mpa.png')

######
plt.show()
