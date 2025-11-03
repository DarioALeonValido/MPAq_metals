import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

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

qmaxGN = 19 #including q=0
dqGN = 1/2/qmaxGN
q_GN = np.linspace(0, 0.5, qmaxGN)


# Plots ----------------------------------------------------------

plt.rcParams['font.size'] = '11.5'
plt.rcParams['lines.linewidth'] = 1.5

f=1.12
fx=1;fy=2
fig3, axs3 = plt.subplots(fy, fx,constrained_layout=True)
fig3.set_size_inches(f*4*fx, f*3*fy/1.1)

#########################################

labels = ['Li', 'Be', 'Na', 'Mg', 'Al',  'K', 
          'Ca', 'Ti',  'V', 'Cr', 'Fe', 'Co', 'Ni', 
          'Cu', 'Zn', 'Mo', 'Pd', 'Ag', 'Sn',
          'Ta' , 'W', 'Os',
                      'Pt', 'Au', 'Tl', 'Pb']
rR = np.array([1.429943295720159213,4.708872470932663035,2.732667190099824950,5.559399229647661755
                                                                            -0.7111625191758995657,
                                                                                                   6.827380289949194214,1.379011571993005614,
               2.086224668427064444,3.950863356652285496,1.628815942767726455,1.815918277293665284,2.107216414718977582,1.411193822074656712,3.925682715194537931,
               1.925721166797522699,1.218286080391839388,3.054052586263733993,5.839128812560550941,0.8692237531442138510,5.156326703404492839,
               4.349854850141371188,5.083614153141936320
                                   +1.247392129649514070,7.442857870645267049
                                                        ,2.392095400323229981,1.578722938628005501,2.103874140251677183,1.881657264651895201])
rE = np.array([6.988000341776142044,19.60060975629115276,5.779763307978837972,10.31571438410222719,14.78569615264617809,3.784500774906751630,
               8.287869482638667407,17.29211451519272202,20.85666086560321730,24.05724531340342409,25.81043535387534860,26.30314267294449238,31.72228041305291057,
               25.80299668863122875,19.06350633098638170,24.16789608000435408,30.28096290524286616,24.23859435739130674,12.84533622388969931,
               21.02347404243108286,24.22148819134242714,32.52081033190461312
                                                        ,32.64918040299929203,23.87189044564524920,9.735719489268545246,12.01962411862426627])

nval = np.array([1,2,1,2,3,1,2,4,5,6,8,9,10,11,12,6,10,11,14,18,19,22,
                                                                   24,25,27,28])
uvol = np.array([3.51**3/2, 2.2858**2*3.5843*3**0.5/2/2, 4.2906**3/2, 3.2094**2*5.2108*3**0.5/2/2, 4.0495**3/4, 5.328**3/2, 
               5.5884**3/4, 2.9508**2*4.6855*3**0.5/2/2,   3.03**3/2,                   2.91**3/2, 2.8665**3/2, 2.5071**2*4.0695*3**0.5/2/2, 3.524**3/4,
               3.6149**3/4, 2.6649**2*4.9468*3**0.5/2/2,  3.147**3/2,                 3.8907**3/4, 4.0853**3/4,          5.8318**2*3.1819/4, 
               3.3013**3/2,                 3.1652**3/2, 2.7344**2*4.3173*3**0.5/2/2, 
                                                         3.9242**3/4, 4.0782**3/4, 3.4566**2*5.5248*3**0.5/2/2, 4.9508**3/4
])
print('uvol',uvol)
conversion = 0.000725246104 #eV^-2 AA^-3

uvol2 = np.array([20.2846, 15.8031/2, 37.1672, 45.7296/2, 4.0495**3/4, 73.3511, 42.1095, 34.7473/2, 13.3709, 11.5493, 11.3353, 21.6318/2, 10.8749, 
          11.9242, 30.3675/2, 15.7658, 15.2902, 17.7762, 28.0011, 18.1599, 16.1103, 28.5276/2,

                                                                                  15.5832, 17.7800, 63.0641/2, 31.6799])

for i in range(len(labels)):
  print(labels[i],nval[i],rE[i]**2*uvol2[i]*conversion)

uvol = uvol2
axs3[0].scatter(nval, rE**2*uvol2*conversion, label='')
axs3[0].plot([0,max(nval)], [0,max(nval)], ls='--', c='k', label='$Z_{\mathrm{eff}} = Z_{\mathrm{val}}$')
axs3[0].set_ylabel("$Z_{\mathrm{eff}}$")
axs3[0].set_xlabel("$Z_{\mathrm{val}}$")
axs3[0].legend(edgecolor='k',loc=2)
axs3[0].set_xlim(-0.9,29.1)
axs3[0].set_ylim(0,14)

# Prepare annotations
texts = []
for i, label in enumerate(labels):
    texts.append(axs3[0].text(nval[i], rE[i]**2*uvol[i]*conversion, label))

# Adjust positions to prevent overlaps
adjust_text(texts, ax=axs3[0], expand_points=(1.2, 1.4), 
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=5, shrinkB=5)
            )


axs3[1].scatter(rE, 2*rR, label='most intense peak')
extra = 1.
axs3[1].plot([min(rE)-extra, (max(rE)+extra)*0.93], [min(rE)-extra, (max(rE)+extra)*0.93], ls='--', c='k',label='$2 R = \Omega$')
axs3[1].plot([min(rE)-extra, max(rE)+extra], [(min(rE)-extra)/2, (max(rE)+extra)/2], ls='--', c='gray',label='$2 R = \Omega$/2')

# Prepare annotations
texts = []
for i, label in enumerate(labels):
    texts.append(axs3[1].text(rE[i], 2 * rR[i], label))

# Adjust positions to prevent overlaps
adjust_text(texts, ax=axs3[1], expand_points=(1.2, 1.4), 
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=5, shrinkB=5)
            )

axs3[0].xaxis.set_minor_locator(AutoMinorLocator(5))
axs3[0].yaxis.set_minor_locator(AutoMinorLocator(2))

axs3[1].xaxis.set_minor_locator(AutoMinorLocator(5))
axs3[1].yaxis.set_minor_locator(AutoMinorLocator(5))

axs3[0].annotate('(a)', (max(nval)*0.95,13),weight='bold')
axs3[1].annotate('(b)', ((max(rE)+extra)*0.955,30),weight='bold')

axs3[1].set_xlabel("Re[$\Omega$] (eV)")
axs3[1].set_ylabel("Re[$2 R$] (eV)")
axs3[1].legend(edgecolor='k')
fig3.savefig('main-peak_all2.pdf')

######
plt.show()