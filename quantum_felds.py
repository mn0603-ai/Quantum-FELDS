# Supplementary Software

"""
Quantum FELDS

Reference implementation accompanying:

Quantum FELDS:
Quantum Probability Densities of Collective
Hydrogen-Bond Network Coordinates in Liquid Water

Author:
Nobuhiro Miura
Ritsumeikan University
"""


# Quantum FELDS analysis code used in the present work.

# The code includes:
# - molecular trajectory analysis
# - free-energy landscape construction
# - Hessian analysis
# - effective-mass evaluation
# - Schrödinger-equation solution
# - projection of quantum probability densities
#


# %% 
# import

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# %%
# theta_pair 読み込み

# theta = np.loadtxt(
#     r"C:\Users\user\water_model\theta_pair.dat"
# )

theta = np.loadtxt(
    "theta_pair.dat"
)


theta_deg = (
    theta*180/np.pi
)

print(
    theta_deg.mean()
)

print(
    theta_deg.min()
)

print(
    theta_deg.max()
)

# %%
# roo 読み込み

# roo = np.loadtxt(
#     r"C:\Users\user\water_model\roo.dat"
# )

roo = np.loadtxt(
    "roo.dat"
)


print(
    roo.mean()
)

print(
    roo.min()
)

print(
    roo.max()
)

# %%
# data check
#

print(
    len(roo),
    len(theta)
)



# %%
# histgram


P2d, r_edges, th_edges = np.histogram2d(
    roo,
    theta,
    bins=[100,100],
    density=True
)

# %%
# plot
#

plt.figure(figsize=(6,5))

plt.imshow(
    P2d.T,
    origin="lower",
    aspect="auto"
)

plt.colorbar(
    label=r"$P$($r_{oo}$,\theta)"
)


plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)



plt.show()

# %%
# PMF

P,bins = np.histogram(
    theta,
    bins=100,
    density=True
)

theta_center = 0.5*(
    bins[:-1]
    +
    bins[1:]
)


P,bins = np.histogram(
    roo,
    bins=100,
    density=True
)

roo_center = 0.5*(
    bins[:-1]
    +
    bins[1:]
)



kB = 0.008314
T = 300.0

F2d = (
    -kB*T
    * np.log(
        P2d + 1e-12
    )
)

F2d -= np.min(F2d)


plt.figure(figsize=(6,5))

plt.contourf(
    roo_center,
    theta_center,
    F2d.T,
    levels=30
)

plt.colorbar(
    label="F(r,θ)"
)

r_center = 0.5*(
    r_edges[:-1]
    +
    r_edges[1:]
)

th_center = 0.5*(
    th_edges[:-1]
    +
    th_edges[1:]
)


plt.figure(figsize=(6,5))

plt.contour(
    r_center,
    th_center,
    F2d.T,
    colors="black",
    linewidths=0.5
)

plt.xlabel("rOO (nm)")
plt.ylabel("theta (rad)")

plt.colorbar(
    label="F(r,θ) [kJ/mol]"
)

plt.show()



# %%
# plot
# 論文図
#

plt.figure(figsize=(6,5))

plt.contourf(
    roo_center,
    theta_center,
    P2d.T,
    levels=30
)


plt.colorbar(
    label=r"$P$ ($r_{oo}$,$\theta$)"

)


plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)



plt.show()







# %%
# PMF 平滑化


F2d_s = gaussian_filter(
    F2d,
    sigma=1.0
)

# %%
# ヘッセ行列

# グリッド間隔
dr = r_center[1] - r_center[0]

dth = (
    th_center[1]
    -
    th_center[0]
)


# 一次微分
Fr, Fth = np.gradient(
    F2d,
    dr,
    dth
)

#  二次微分

Frr, Frth = np.gradient(
    Fr,
    dr,
    dth
)

Fthr, Fthth = np.gradient(
    Fth,
    dr,
    dth
)

# 結合項の可視化

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    Frth.T,
    levels=30
)

plt.colorbar(
    label=r'$\partial^2F/\partial r\partial\theta$'
)

plt.xlabel("rOO (nm)")
plt.ylabel("theta (rad)")

plt.show()


# %%
# PMF極小でヘッセ行列

imin, jmin = np.unravel_index(
    np.argmin(F2d),
    F2d.shape
)

print(
    "imin =", imin
)

print(
    "jmin =", jmin
)

print(
    "r* =",
    r_center[imin]
)

print(
    "theta* =",
    th_center[jmin]
)

H = np.array([
    [
        Frr[imin,jmin],
        Frth[imin,jmin]
    ],
    [
        Fthr[imin,jmin],
        Fthth[imin,jmin]
    ]
])

print(H)

eigvals, eigvecs = np.linalg.eigh(H)

print(
    "eigenvalues"
)

print(
    eigvals
)

print()

print(
    "eigenvectors"
)

print(
    eigvecs
)

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    F2d.T,
    levels=30,
    cmap="viridis"
)

plt.colorbar(
    label="F(r,θ)"
)

plt.plot(
    r_center[imin],
    th_center[jmin],
    "ro",
    markersize=10
)

plt.xlabel("rOO (nm)")
plt.ylabel("theta (rad)")

plt.show()


# %%

coupling = np.abs(Frth)

print(
    np.max(coupling)
)

print(
    np.mean(coupling)
)



plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    np.abs(Frth).T,
    levels=30
)

plt.colorbar(
    label="F(r,θ)"
)

plt.plot(
    r_center[imin],
    th_center[jmin],
    "ro",
    markersize=10
)

plt.xlabel("rOO (nm)")
plt.ylabel("theta (rad)")

plt.show()




# %%
# 極小からの変位を作る
#


r_star = r_center[imin]
theta_star = th_center[jmin]



dr = roo - r_star
dth = theta - theta_star

# %%
# 主軸座標に変換

R = eigvecs

Q1 = (
    R[0,0]*dr
    + R[1,0]*dth
)

Q2 = (
    R[0,1]*dr
    + R[1,1]*dth
)


# %%
# 主軸確認
#

plt.figure(figsize=(6,4))

plt.scatter(
    Q1,
    Q2,
    s=1,
    alpha=0.2
)

plt.xlabel("Q1")
plt.ylabel("Q2")

plt.show()


# %%
# ヒストグラム　P(Q1, Q2)
#

P2d_Q, Q1_edges, Q2_edges = np.histogram2d(
    Q1,
    Q2,
    bins=[100,100],
    density=True
)

plt.figure(figsize=(6,5))

plt.imshow(
    P2d_Q.T,
    origin="lower",
    aspect="auto"
)

plt.colorbar(
    label="P(Q1,Q2)"
)

plt.show()



# %%
# PMF
#

P,bins = np.histogram(
    Q1,
    bins=100,
    density=True
)

Q1_center = 0.5*(
    bins[:-1]
    +
    bins[1:]
)


P,bins = np.histogram(
    Q2,
    bins=100,
    density=True
)

Q2_center = 0.5*(
    bins[:-1]
    +
    bins[1:]
)



kB = 0.008314
T = 300.0

F2d_Q = (
    -kB*T
    * np.log(
        P2d_Q + 1e-12
    )
)

F2d -= np.min(F2d_Q)

# %%
# PMF 論文図


plt.figure(figsize=(6,5))


plt.contourf(
    Q1_center,
    Q2_center,
    F2d_Q.T,
    levels=30,
    cmap="viridis"
)

plt.plot(
    0.0,
    0.0,
    "wo",
    ms=10,
    mec="k"
)


scale = 0.3

plt.quiver(
    0,
    0,
    eigvecs[0,0],
    eigvecs[1,0],
    color="white",
    scale=1/scale
)

plt.quiver(
    0,
    0,
    eigvecs[0,1],
    eigvecs[1,1],
    color="red",
    scale=1/scale
)


plt.text(
    0.10,
    -0.03,
    r"$Q_2$",
    color="white",
    fontsize=14
)

plt.text(
    -0.5,
    0.01,
    r"$Q_1$",
    color="red",
    fontsize=14
)


plt.colorbar(
    label="$F (Q_1,Q_2)$ [kJ / mol]"
)


plt.xlabel("$Q_1$")
plt.ylabel("$Q_2$")

plt.show()


# %%

plt.figure(figsize=(6,5))

plt.contour(
    Q1_center,
    Q2_center,
    F2d_Q.T,
    colors="black",
    linewidths=0.5
)

plt.xlabel("Q1")
plt.ylabel("Q2")

plt.colorbar(
    label="F (Q1,Q2) [kJ/mol]"
)

plt.show()





# 
# 1次元のシュレディンガー方程式
#　Q1
#
#


# %%
# F(Q1)の一次元分布
#


kB = 0.008314
T = 300.0


P,bins = np.histogram(
    Q1,
    bins=100,
    density=True
)

Q1_center = 0.5*(
    bins[:-1]
    +
    bins[1:]
)

F_Q1 = (
    -kB*T
    * np.log(
        P + 1e-12
    )
)

F_Q1 -= np.min(F_Q1)



# %%
# plot
#


plt.figure(figsize=(6,4))

plt.plot(
    Q1_center,
    F_Q1,
    "r-"
)


plt.ylabel("F_Q1 [kJ/mol]")
plt.xlabel("Q1")

plt.show()


# %%
# SI単位に変換
#

V = F_Q1.copy()

V -= np.min(V)

NA = 6.02214076e23

V = (
    V*1000.0
    / NA
)

# %%
# 格子化
#

dq = (
    Q1_center[1]
    -
    Q1_center[0]
)

N = len(Q1_center)


# %%
# check
#

print(
    np.min(Q1),
    np.max(Q1)
)

print(
    np.min(F_Q1),
    np.max(F_Q1)
)


#
#
# M_effを求めてシュレディンガー方程式を解く
#
#

# %%
# Q1の時間微分
#

dt = 1.0e-14

Q1_dot = np.gradient(
    Q1,
    dt
)

plt.figure(figsize=(6,4))

plt.plot(
    Q1_dot[:10000]
)

plt.ylabel("dQ1/dt")
plt.xlabel("step")

plt.show()

# %%
# ヒストグラム

plt.figure(figsize=(6,4))

plt.hist(
    Q1_dot,
    bins=100,
    density=True
)

plt.xlabel("dQ1/dt")

plt.show()


# %%
# 平均二乗速度
#

Q1dot2 = np.mean(
    Q1_dot**2
)

print(Q1dot2)

# %%
# 換算質量
#

kB = 1.380649e-23

T = 300.0

M_eff = (
    kB*T
    /
    Q1dot2
)

print(M_eff)



# %%
# potential
#

NA = 6.02214076e23

V = F_Q1.copy()

V -= np.min(V)

V = (
    V*1000.0
    / NA
)

# %%
# kinetic energy
#

hbar = 1.054571817e-34

prefactor = (
    hbar**2
    /
    (2*M_eff*dq**2)
)

# %%
# Hamiltonian
#

diag = np.full(
    N,
    2.0*prefactor
)

offdiag = np.full(
    N-1,
    -prefactor
)

H = np.diag(diag)

H += np.diag(
    offdiag,
    1
)

H += np.diag(
    offdiag,
    -1
)

H += np.diag(V)


# %%
# 固有値問題
#

E, psi = np.linalg.eigh(H)


# %%
# 波動関数の規格化
#

for n in range(4):

    norm = np.trapezoid(
        psi[:,n]**2,
        Q1_center
    )

    psi[:,n] /= np.sqrt(
        norm
    )

# %%
# check
#

for n in range(4):

    print(
        n,
        np.trapezoid(
            psi[:,n]**2,
            Q1_center
        )
    )



# %%
# plot of ψ
#



plt.figure(figsize=(6,4))

plt.plot(
    Q1_center,
    psi[:,0],
    label="Ground state"
    )

plt.plot(
    Q1_center,
    psi[:,1],
    label="First excited state"
)

plt.plot(
    Q1_center,
    psi[:,2],
    label="second excited state"
)

plt.plot(
    Q1_center,
    psi[:,3],
    label="third excited state"
)


plt.xlabel(
    r"$Q_1$",
    fontsize=14
)

plt.ylabel(
    r"$\psi$",
    fontsize=14
)



plt.show()


# %%
# |ψ|²
# 論文図
#

plt.figure(figsize=(6,4))

for n in range(4):

    plt.plot(
        Q1_center,
        psi[:,n]**2,
        label=f"{n}"
    )


plt.xlabel(r"$Q_1$",
           fontsize=14)

plt.ylabel(r"$|\psi_n(Q_1)|^2$",
           fontsize=14
           )


# plt.legend()

plt.show()


# %%
# エネルギー準位
# 論文図
#

NA = 6.02214076e23

plt.figure(figsize=(6,4))

plt.plot(
    Q1_center,
    F_Q1,
    color="black"
)

for n in range(6):

    E_kJ = (
        E[n]
        * NA
        /1000
    )

    plt.axhline(
        E_kJ,
        ls="--"
    )

plt.ylim(-2,22)

# plt.xlabel("Q2")
plt.xlabel(
    r"$Q_1$",
    fontsize=14
)


plt.tick_params(
    direction="in",
    top=True,
    right=True,
    length=6
)


plt.tick_params(
    which="minor",
    direction="in",
    top=True,
    right=True,
    length=3
)

plt.grid(False)

plt.ylabel("$F(Q_1)$ [kJ / mol]",
           fontsize=14
           )


plt.show()



# %%
# check

print(
    Q1_center.min(),
    Q1_center.max()
)

print(
    dq
)

print(
    np.min(V),
    np.max(V)
)

print(E[:10])


# %%
# check

for i in range(10):

    print(
        i,
        E[i]
    )

for i in range(5):

    print(
        i,
        (E[i]-E[0])
    )

for n in range(4):

    norm = np.trapezoid(
        psi[:,n]**2,
        Q1_center
    )
    print(
        n,
        norm  )



# %%
# 遷移周波数
#

c = 2.99792458e10

nu01 = (
    (E[1]-E[0])
    /
    (6.62607015e-34*c)
)

print(nu01)

# %%
# check

print(Q1.min(),Q1.max())
print(dq)


print(M_eff)




# %%
# 実空間射影 Q1
#

# 基底状態の波動関数

psi0_sq = psi[:,0]**2

# 重み　w

from scipy.interpolate import interp1d

fpsi = interp1d(
    Q1_center,
    psi[:,0]**2,
    bounds_error=False,
    fill_value=0.0
)

w = fpsi(Q1)

# %%
# histgram
#

P0, r_edges, th_edges = np.histogram2d(
    roo,
    theta,
    bins=[100,100],
    weights=w,
    density=True
)

# %%
# plot
#

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    P0.T,
    levels=30
)

plt.colorbar(
    label="Ground-state density"
)


plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)

plt.show()



# %%
# 励起状態の波動関数 first
#

psi0_sq = psi[:,1]**2

# 重み　w

from scipy.interpolate import interp1d

fpsi = interp1d(
    Q1_center,
    psi[:,1]**2,
    bounds_error=False,
    fill_value=0.0
)

w = fpsi(Q1)

# %%
# histgram
#

P1, r_edges, th_edges = np.histogram2d(
    roo,
    theta,
    bins=[100,100],
    weights=w,
    density=True
)

# %%
# plot
#

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    P1.T,
    levels=30
)

plt.colorbar(
    label="First-excited-state density"
)


plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)


plt.show()




# %%
# 励起状態の波動関数 Second
#

# psi0_sq = psi[:,2]**2

# 重み　w

from scipy.interpolate import interp1d

fpsi = interp1d(
    Q1_center,
    psi[:,2]**2,
    bounds_error=False,
    fill_value=0.0
)

w = fpsi(Q1)

# %%
# histgram
#

P2, r_edges, th_edges = np.histogram2d(
    roo,
    theta,
    bins=[100,100],
    weights=w,
    density=True
)

# %%
# plot
#

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    P2.T,
    levels=30
)

plt.colorbar(
    label="Second-excited-state density"
)


plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)


plt.show()



# %%
# 励起状態の波動関数 Third
#

# psi0_sq = psi[:,2]**2

# 重み　w

from scipy.interpolate import interp1d

fpsi = interp1d(
    Q1_center,
    psi[:,3]**2,
    bounds_error=False,
    fill_value=0.0
)

w = fpsi(Q1)

# %%
# histgram
#

P3, r_edges, th_edges = np.histogram2d(
    roo,
    theta,
    bins=[100,100],
    weights=w,
    density=True
)

# %%
# plot
#

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    P3.T,
    levels=30
)

plt.colorbar(
    label="Third-excited-state density"
)

plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)

plt.show()









# 
# 1次元のシュレディンガー方程式
#　Q2
#
#


# %%
# F(Q2)の一次元分布
#


kB = 0.008314
T = 300.0


P,bins = np.histogram(
    Q2,
    bins=100,
    density=True
)

Q2_center = 0.5*(
    bins[:-1]
    +
    bins[1:]
)

F_Q2 = (
    -kB*T
    * np.log(
        P + 1e-12
    )
)

F_Q2 -= np.min(F_Q2)



# %%
# plot
#

plt.figure(figsize=(6,4))

plt.plot(
    Q2_center,
    F_Q2,
    "r-"
)


plt.ylabel("F_Q2 [kJ/mol]")
plt.xlabel("Q2")

plt.show()


# %%
# SI単位に変換
#

V = F_Q2.copy()

V -= np.min(V)

NA = 6.02214076e23

V = (
    V*1000.0
    / NA
)

# %%
# 格子化
#

dq = (
    Q2_center[1]
    -
    Q2_center[0]
)

N = len(Q2_center)


# %%
# check
#

print(
    np.min(Q2),
    np.max(Q2)
)

print(
    np.min(F_Q2),
    np.max(F_Q2)
)


#
#
# M_effを求めてシュレディンガー方程式を解く
#
#

# %%
# Q2の時間微分
#

dt = 1.0e-14

Q2_dot = np.gradient(
    Q2,
    dt
)

plt.figure(figsize=(6,4))

plt.plot(
    Q2_dot[:10000]
)

plt.ylabel("dQ2/dt")
plt.xlabel("step")

plt.show()

# %%
# ヒストグラム

plt.figure(figsize=(6,4))

plt.hist(
    Q2_dot,
    bins=100,
    density=True
)

plt.xlabel("dQ2/dt")

plt.show()


# %%
# 平均二乗速度
#

Q2dot2 = np.mean(
    Q2_dot**2
)

print(Q2dot2)

# %%
# 換算質量
#

kB = 1.380649e-23

T = 300.0

M_eff = (
    kB*T
    /
    Q2dot2
)

print(M_eff)



# %%
# potential
#

NA = 6.02214076e23

V = F_Q2.copy()

V -= np.min(V)

V = (
    V*1000.0
    / NA
)

# %%
# kinetic energy
#

hbar = 1.054571817e-34

prefactor = (
    hbar**2
    /
    (2*M_eff*dq**2)
)

# %%
# Hamiltonian
#

diag = np.full(
    N,
    2.0*prefactor
)

offdiag = np.full(
    N-1,
    -prefactor
)

H = np.diag(diag)

H += np.diag(
    offdiag,
    1
)

H += np.diag(
    offdiag,
    -1
)

H += np.diag(V)


# %%
# 固有値問題
#

E, psi = np.linalg.eigh(H)


# %%
# 波動関数の規格化
#

for n in range(4):

    norm = np.trapezoid(
        psi[:,n]**2,
        Q2_center
    )

    psi[:,n] /= np.sqrt(
        norm
    )

# %%
# check
#

for n in range(4):

    print(
        n,
        np.trapezoid(
            psi[:,n]**2,
            Q2_center
        )
    )



# %%
# plot of ψ
#



plt.figure(figsize=(6,4))

plt.plot(
    Q2_center,
    psi[:,0],
    label="Ground state"
    )

plt.plot(
    Q2_center,
    psi[:,1],
    label="First excited state"
)

plt.plot(
    Q2_center,
    psi[:,2],
    label="Second excited state"
)

plt.plot(
    Q2_center,
    psi[:,3],
    label="Third excited state"
)

plt.xlabel(
    r"$Q_2$",
    fontsize=14
)

plt.ylabel(
    r"$\psi$",
    fontsize=14
)

plt.legend()

plt.show()



# %%
# |ψ|²
#

plt.figure(figsize=(6,4))

for n in range(4):

    plt.plot(
        Q2_center,
        psi[:,n]**2,
        label=f"{n}"
    )


plt.xlabel(
    r"$Q_2$",
    fontsize=14
)

plt.ylabel(
    r"$\psi$",
    fontsize=14
)


plt.show()



# %%
# |ψ|²
# 論文図
#

plt.figure(figsize=(6,4))

plt.plot(
    Q2_center,
    psi[:,0]**2,
    label="Ground state"
    )

plt.plot(
    Q2_center,
    psi[:,1]**2,
    label="First excited state"
)

plt.plot(
    Q2_center,
    psi[:,2]**2,
    label="Second excited state"
)

plt.plot(
    Q2_center,
    psi[:,3]**2,
    label="Third excited state"
)


plt.xlabel(
    r"$Q_2$",
    fontsize=14
)

plt.ylabel(r"$|\psi_n(Q_1)|^2$",
           fontsize=14
           )



plt.legend()


plt.show()





# %%
# エネルギー準位
# 論文図
#

NA = 6.02214076e23

plt.figure(figsize=(6,4))

plt.plot(
    Q2_center,
    F_Q2,
    color="black"
)

for n in range(6):

    E_kJ = (
        E[n]
        * NA
        /1000
    )

    plt.axhline(
        E_kJ,
        ls="--"
    )

plt.ylim(-2,36)

# plt.xlabel("Q2")
plt.xlabel(
    r"$Q_2$",
    fontsize=14
)


plt.tick_params(
    direction="in",
    top=True,
    right=True,
    length=6
)


plt.tick_params(
    which="minor",
    direction="in",
    top=True,
    right=True,
    length=3
)

plt.grid(False)

plt.ylabel("$F(Q_2)$ [kJ / mol]",
           fontsize=14
           )

plt.show()



# %%
# check

print(
    Q2_center.min(),
    Q2_center.max()
)

print(
    dq
)

print(
    np.min(V),
    np.max(V)
)

print(E[:10])


# %%
# check

for i in range(10):

    print(
        i,
        E[i]
    )

for i in range(5):

    print(
        i,
        (E[i]-E[0])
    )

for n in range(4):

    norm = np.trapezoid(
        psi[:,n]**2,
        Q2_center
    )
    print(
        n,
        norm  )



# %%
# 遷移周波数
#

c = 2.99792458e10

nu01 = (
    (E[1]-E[0])
    /
    (6.62607015e-34*c)
)

print(nu01)

# %%
# check

print(Q2.min(),Q2.max())
print(dq)


print(M_eff)




# %%
# 実空間射影 Q2
#

# 基底状態の波動関数

psi0_sq = psi[:,0]**2

# 重み　w

from scipy.interpolate import interp1d

fpsi = interp1d(
    Q2_center,
    psi[:,0]**2,
    bounds_error=False,
    fill_value=0.0
)

w = fpsi(Q2)

# %%
# histgram
#

P0, r_edges, th_edges = np.histogram2d(
    roo,
    theta,
    bins=[100,100],
    weights=w,
    density=True
)

# %%
# plot
#

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    P0.T,
    levels=30
)

# plt.title(
#     r"Quantum-State Density Along $Q_2$"
# )


plt.colorbar(
    label="Grand-state density"
)

plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)

plt.show()



# %%
# 実空間射影 Q2
#
#
# 励起状態の波動関数

# psi0_sq = psi[:,1]**2

# 重み　w

from scipy.interpolate import interp1d

fpsi = interp1d(
    Q2_center,
    psi[:,1]**2,
    bounds_error=False,
    fill_value=0.0
)

w = fpsi(Q2)

# %%
# histgram
#

P1, r_edges, th_edges = np.histogram2d(
    roo,
    theta,
    bins=[100,100],
    weights=w,
    density=True
)

# %%
# plot
#

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    P1.T,
    levels=30
)

plt.colorbar(
    label="First-excited-state density"
)


plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)


plt.show()




# %%
# 第２励起状態の波動関数

fpsi = interp1d(
    Q2_center,
    psi[:,2]**2,
    bounds_error=False,
    fill_value=0.0
)

w = fpsi(Q2)

# %%
# histgram
#

P2, r_edges, th_edges = np.histogram2d(
    roo,
    theta,
    bins=[100,100],
    weights=w,
    density=True
)

# %%
# plot
#

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    P2.T,
    levels=30
)

plt.colorbar(
    label="Second-excited-state density"
)


plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)

plt.show()



# %%
# 第3励起状態の波動関数


from scipy.interpolate import interp1d

fpsi = interp1d(
    Q2_center,
    psi[:,3]**2,
    bounds_error=False,
    fill_value=0.0
)

w = fpsi(Q2)

# %%
# histgram
#

P3, r_edges, th_edges = np.histogram2d(
    roo,
    theta,
    bins=[100,100],
    weights=w,
    density=True
)

# %%
# plot
#

plt.figure(figsize=(6,5))

plt.contourf(
    r_center,
    th_center,
    P3.T,
    levels=30
)

plt.colorbar(
    label="Third-excited-state density"
)


plt.xlabel(r"$r_{OO}$ [nm]",
           fontsize=14)

plt.ylabel(r"$\theta$ [rad]",
           fontsize=14)

plt.show()


# %%
