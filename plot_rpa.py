import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d


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
    eel_interp = f_q(q_dense)

    return q_dense, eel_interp


def build_q_path(lattice_type="bcc"):
    """
    Return q_list (file indices), q_path (x-axis positions),
    xticks (tick positions + labels), and gamma_idx (index of Γ in q_path).
    """
    gamma = 0.0

    if lattice_type == "bcc":
        # -------- BCC path: H -> Γ -> N --------
        q_points = np.array([145,359,539,689,811,909,987,1047,1093,1130,1184,1217,1227,1234,1240])
        qmaxGN = 19
        q_GN = np.linspace(0, 0.5, qmaxGN)

        qmax1 = len(q_points) + 1
        q_1 = np.linspace(-0.5 * qmax1 / qmaxGN, 0, qmax1)
        q_list = np.concatenate((q_points[::-1], range(2, qmaxGN + 1)))
        q_path = np.concatenate((q_1[:-1], q_GN[1:]))

        xticks = ([-0.5 * qmax1 / qmaxGN, 0, 0.5], ["$H$", "$\\Gamma$", "$N$"])

    elif lattice_type == "fcc":
        # -------- FCC path: X -> Γ -> N --------
        q_points = np.array([55,88,119,148,175,200,223,244,263,280,295,308,319,328,335,343])
        qmaxGN = 19
        q_GN = np.linspace(0, 0.5, qmaxGN)

        qmax1 = len(q_points) + 1
        q_1 = np.linspace(-0.5 * qmax1 / qmaxGN, 0, qmax1)
        q_list = np.concatenate((q_points[::-1], range(2, qmaxGN + 1)))
        q_path = np.concatenate((q_1[:-1], q_GN[1:]))

        xticks = ([-0.5 * qmax1 / qmaxGN, 0, 0.5], ["$X$", "$\\Gamma$", "$N$"])

    elif lattice_type == "hcp":
        # -------- Hexagonal path: H -> A -> Γ -> M, then K -> Γ --------
        q_points_GM = np.array([14,27,40,53,66,79,92,105,118,131,
                                144,157,170,183,196,209,222,235])   # Γ → M
        q_points_GK = np.array([248,469,677,859,1028,1171,1301,
                                1405,1496,1561,1613,1639])          # Γ → K
        q_points_AH = np.array([260,481,689,871,1040,1183,1313,
                                1417,1508,1573,1625,1651])          # A → H

        qmaxGA = 13  # including q=0
        qmaxGK = len(q_points_GK) + 1
        qmaxAH = len(q_points_AH) + 1
        qmaxGM = len(q_points_GM) + 1

        q_1 = np.linspace(-0.5*qmaxAH/qmaxGA, 0, qmaxAH)
        q_2 = np.linspace(0, 0.5, qmaxGA)
        q_3 = np.linspace(0.5, 1*(qmaxGM/qmaxGA-0.0001), qmaxGM)
        q_4 = np.linspace(1*(qmaxGM/qmaxGA+0.0001),
                          1*(qmaxGM/qmaxGA+0.0001) + 0.5*qmaxGK/qmaxGA, qmaxGK)
        q_5 = np.linspace(1*(qmaxGM/qmaxGA+0.0001) + 0.5*qmaxGK/qmaxGA,
                          1*(qmaxGM/qmaxGA+0.0001) + 0.5*qmaxGK/qmaxGA + 0.5*qmaxGK/qmaxGA, qmaxGK)

        q_list = np.concatenate((q_points_AH[::-1], range(qmaxGA,1,-1), q_points_GM,
                                 q_points_GK[::-1], q_points_GK[:2]))
        q_path = np.concatenate((q_1[:-1], q_2[:-1], q_3[1:], q_4[:-1], q_5[:2]))

        xticks = ([-0.5*qmaxAH/qmaxGA, 0, 0.5,
                   1*qmaxGM/qmaxGA,
                   1*(qmaxGM/qmaxGA+0.0001) + 0.5*qmaxGK/qmaxGA],
                  ["$H$", "$A$", "$\\Gamma$", "$M$ | $K$", "$\\Gamma$"])
        gamma = 1*(qmaxGM/qmaxGA-0.0001) + 0.5*qmaxGK/qmaxGA

    elif lattice_type == "Sn_cubic":
        # -------- Special Sn path: L -> Γ -> X --------
        q_points = np.array([38,57,75,94,112,131,149,168,186,205,223])
        qmaxGX = 19
        q_GX = np.linspace(0, 0.5, qmaxGX)

        qmax1 = len(q_points) + 1
        q_1 = np.linspace(-0.5 * qmax1 / qmaxGX, 0, qmax1)
        q_list = np.concatenate((q_points[::-1], range(2, qmaxGX + 1)))
        q_path = np.concatenate((q_1[:-1], q_GX[1:]))

        xticks = ([-0.5 * qmax1 / qmaxGX, 0, 0.5], ["$Z$", "$\\Gamma$", "$X$"])

    else:
        raise ValueError("lattice_type must be 'bcc', 'fcc', 'hcp', or 'Sn_cubic'")

    # --- locate Γ index ---
    gamma_idx = int(np.argmin(np.abs(q_path - gamma)))

    return q_list, q_path, xticks, gamma_idx

#########################################################################################

materials = {
    "Li": {"folder": "Li_k36", "symm": "bcc", "refs": {"eels": None,                          "reels": None,     "ip": None}},
    "Be": {"folder": "Be_k36", "symm": "hcp", "refs": {"eels": "Be(0-0180eV)_nZLP.txt",       "reels": None,     "ip": None}},
    "Na": {"folder": "Na_k36", "symm": "bcc", "refs": {"eels": None,                          "reels": None,     "ip": None}},
    "Mg": {"folder": "Mg_k36", "symm": "hcp", "refs": {"eels": None,                          "reels": None,     "ip": None}},
    "Al": {"folder": "Al_k36", "symm": "fcc", "refs": {"eels": None,       "reels": None,     "ip": None}},
    "K":  {"folder": "K_k36",  "symm": "bcc", "refs": {"eels": None,                          "reels": None,     "ip": None}},
    "Ca": {"folder": "Ca_k36", "symm": "fcc", "refs": {"eels": None,                          "reels": None,     "ip": None}},
    "Ti": {"folder": "Ti_k36", "symm": "hcp", "refs": {"eels": None,                          "reels": "Ti.dat",  "ip": "Ti.dat"}},
    "V":  {"folder": "V_k36",  "symm": "bcc", "refs": {"eels": "V(0-0180eV)_nZLP.txt",        "reels": "V.dat",   "ip": "V.dat"}},
    "Cr": {"folder": "Cr_k36", "symm": "bcc", "refs": {"eels": "Cr(0-0180eV)_nZLP.txt",       "reels": None,     "ip": None}},
    "Fe": {"folder": "Fe_k36", "symm": "bcc", "refs": {"eels": None,                          "reels": "Fe.dat",  "ip": "Fe.dat"}},
    "Co": {"folder": "Co_k36", "symm": "hcp", "refs": {"eels": None,                          "reels": "Co.dat",  "ip": "Co.dat"}},
    "Ni": {"folder": "Ni_k36", "symm": "fcc", "refs": {"eels": None,                          "reels": "Ni.dat",  "ip": "Ni.dat"}},
    "Cu": {"folder": "Cu_k36", "symm": "fcc", "refs": {"eels": "Cu(0-0180eV)_nZLP.txt",       "reels": "Cu.dat",  "ip": "Cu.dat"}},
    "Zn": {"folder": "Zn_k36", "symm": "hcp", "refs": {"eels": "Zn(0-0170eV)_nZLP.txt",       "reels": "Zn.dat",  "ip": "Zn.dat"}},
    "Mo": {"folder": "Mo_k36", "symm": "bcc", "refs": {"eels": None,                          "reels": "Mo.dat",  "ip": "Mo.dat"}},
    "Pd": {"folder": "Pd_k36", "symm": "fcc", "refs": {"eels": None,                          "reels": "Pd.dat",  "ip": "Pd.dat"}},
    "Ag": {"folder": "Ag_k36", "symm": "fcc", "refs": {"eels": "Ag(0-0180eV)_nZLP.txt",       "reels": "Ag.dat",  "ip": "Ag.dat"}},
    "Sn": {"folder": "Sn_k36", "symm": "Sn_cubic", "refs": {"eels": "Sn(0-0180eV)_nZLP.txt", "reels": None,     "ip": None}},
    "Ta": {"folder": "Ta_k36", "symm": "bcc", "refs": {"eels": None,                          "reels": "Ta.dat",  "ip": "Ta.dat"}},
    "W":  {"folder": "W_k36",  "symm": "bcc", "refs": {"eels": "W(0-0180eV)_nZLP.txt",        "reels": "W.dat",   "ip": "W.dat"}},
    "Os": {"folder": "Os_k36", "symm": "hcp", "refs": {"eels": "Os(0-0190eV)_nZLP.txt",       "reels": None,     "ip": None}},
    "Pt": {"folder": "Pt_k36", "symm": "fcc", "refs": {"eels": None,                          "reels": "Pt.dat",  "ip": "Pt.dat"}},
    "Au": {"folder": "Au_k36", "symm": "fcc", "refs": {"eels": "Au(0-0180eV)_nZLP.txt",       "reels": "Au.dat",  "ip": "Au.dat"}},
    "Tl": {"folder": "Tl_k36", "symm": "hcp", "refs": {"eels": "Tl(0-0180eV)_nZLP.txt",       "reels": None,     "ip": None}},
    "Pb": {"folder": "Pb_k36", "symm": "fcc", "refs": {"eels": "Pb(0-0180eV)_nZLP.txt",       "reels": "Pb.dat",  "ip": "Pb.dat"}},
    "Bi": {"folder": "Bi_k36", "symm": "Bi_cubic", "refs": {"eels": "Bi(0-0180eV)_nZLP.txt","reels": "Bi.dat",  "ip": "Bi.dat"}},
}

# ---------------- User choice ----------------
element = "Os"
mat = materials[element]["folder"]
symm = materials[element]["symm"]
Emax = 100

q_list, q_path, xticks, gamma_idx = build_q_path(symm)

# ---------------- File naming ----------------
out_ha_q = 'o-opt-HA_b500_x5Ry_r0-200eV_d0.1eV_f2000.'
eel = 'eel_q'
theoHA = '_inv_rpa_dyson'

# ---------------- Plot settings ----------------
plt.rcParams['font.size'] = 11.5
plt.rcParams['lines.linewidth'] = 1.5

c_map = "BrBG"
dq_fine = 0.01

# ---------------- Data processing ----------------
map_eel_i = []

for q in q_list:
    energy, eel_i, eel_r = np.genfromtxt(
        f"RPA_data/{mat}/{out_ha_q}{eel}{q}{theoHA}",
        autostrip=True, usecols=[0, 1, 2], unpack=True
    )
    egrid, eel_i_interp = interpolate_spectrum(energy, eel_i, Emax, 0)
    map_eel_i.append(eel_i_interp)

q_dense, map_eel_i = interpolate_dispersion(egrid, map_eel_i, q_path, dq_fine)
map_eel_i = np.array(map_eel_i).T
grid_q, grid_w = np.meshgrid(q_dense, egrid)

# ---------------- Plots ----------------
f=0.75
fig1, ax1 = plt.subplots(figsize=(f*4, f*6),layout='constrained')
if symm == 'hcp':
  fig2, ax2 = plt.subplots(figsize=(f*3*5/3, f*6),layout='constrained')
  pcm = ax2.pcolor(grid_q, grid_w, map_eel_i, cmap=c_map)
  fig2.colorbar(pcm, ax=ax2, location='top', extend='max', aspect=20*5/3)
else:
  fig2, ax2 = plt.subplots(figsize=(f*3, f*6),layout='constrained')
  pcm = ax2.pcolor(grid_q, grid_w, map_eel_i, cmap=c_map)
  fig2.colorbar(pcm, ax=ax2, location='top', extend='max', aspect=20)

tick_pos, tick_labels = xticks
ax2.set_xticks(tick_pos, labels=tick_labels)
for x in tick_pos[1:-1]:
  ax2.plot([x, x], [0, Emax], c='k')

ax2.set_xlabel("$\\mathbf{q}$")
ax2.set_ylabel("$\\omega$ (eV)")
ax2.set_title(mat[:-4], weight='bold')

ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
fig2.savefig('cmap_'+mat[:-4]+'_'+c_map+'.png')

### fig1 ###
Xexp = 'Exp_data/'
energy,eel_i,eel_r = np.genfromtxt('RPA_data/'+mat+'/'+out_ha_q+'eel_q'+str(2)+theoHA,autostrip=True,usecols=[0,1,2],unpack=True)
scale=max(eel_i)
ax1.plot(energy,eel_i,c='tab:blue',label='RPA')

refs = materials[element]["refs"]
if refs["eels"] != None:
    ene,ex_i = np.genfromtxt(Xexp+refs["eels"],delimiter=' ',comments='#',autostrip=True,unpack=True)
    rescale=scale/max(ex_i) 
    ax1.plot(ene,ex_i*rescale,c='k',label='EELS')
if refs["reels"] != None:
    print(refs["reels"])
    ene,dft,reel = np.genfromtxt(Xexp+refs["reels"],comments='#',autostrip=True,unpack=True,usecols=[0,3,7])
    ax1.plot(ene,reel,c='orange',label='REELS')
    ax1.plot(ene,dft,c='r',label='IPA Ref',ls=':')
    scale = max(scale,max(reel),max(dft))

ax1.set_xlabel("$\\omega$ (eV)")
ax1.set_ylabel("$-Im[\epsilon^{-1}]~(a.u.)$")
ax1.set_title(mat[:-4], weight='bold')
ax1.set_xlim(0,100)
ax1.set_ylim(0,scale*1.02)
ax1.legend(edgecolor='k',loc=1)
fig1.savefig('RPA-Exp_'+mat[:-4]+'.png')

############
plt.show()