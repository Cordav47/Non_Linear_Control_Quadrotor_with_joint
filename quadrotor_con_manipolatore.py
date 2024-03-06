import numpy as np
import control as ctrl
import scipy as sc
import math as mt
import matplotlib.pyplot as pl
import sympy as sp
import matplotlib.animation as man
from numpy import sin, cos, array, pi, zeros, matmul, transpose, matmul
from matplotlib.patches import Rectangle, Ellipse
from numpy.linalg import inv
from scipy import io
from random import uniform

#variabili fisse 
m0 = 1
J0 = 0.015
dg = 0.1
g = 9.81 
d1 = 0.1
mlink = 0.2
mmot = 0.05
Jlink = 0.001
Jmot = 0.0003    

#notazione: p significa derivat prima, 2p derivata seconda, 3p derivata terza, etc sempre riferita rispetto al tempo

#tempo
step = 0.01
t = np.arange(0, 15.01, step)
nstep = np.size(t)

#inizializo il vettore traiettorie desiderate 
xd = zeros(nstep)
x2pd = zeros(nstep)
x3pd = zeros(nstep)
x4pd = zeros(nstep)
xpd = zeros(nstep)
zd = zeros(nstep)
zpd = zeros(nstep)
z2pd = zeros(nstep)
z3pd = zeros(nstep)
z4pd = zeros(nstep)
t1d = zeros(nstep)
t1pd = zeros(nstep)
t12pd = zeros(nstep)
t13pd = zeros(nstep)
t14pd = zeros(nstep)

#prendo i dati di traiettoria da matlab, fino alla derivata quarta
matdata = sc.io.loadmat('dati_2.mat')
dati = matdata['vtot']
#la lettera d identifica i dati della traiettoria
xd[:] = dati[0,:] 
zd[:] = dati[1,:] 
t1d[:] = dati[2,:] 
xpd[:] = dati[3,:] 
zpd[:] = dati[4,:] 
t1pd[:] = dati[5,:] 
x2pd[:] = dati[6,:] 
z2pd[:] = dati[7,:] 
t12pd[:] = dati[8,:] 
x3pd[:] = dati[9,:] 
z3pd[:] = dati[10,:] 
t13pd[:] = dati[11,:] 
x4pd[:] = dati[12,:] 
z4pd[:] = dati[13,:] 
t14pd[:] = dati[14,:] 

#la funzione che deriva il sistema e che andrà in pasto all'integratore
def sisder(t, S, K, traj): 
    #t è il tempo, S è il vettore di stato, K i coefficienti di controllo e traj i dati della traiettoria
    #du/dt = w = [utp, uri, taup]
    #dw/dt = ub = [utpp, ur, taupp]
    ut = S[0]
    tau = S[1]
    utp= S[2]
    taup = S[3]
    #dy/dt = yp = [xp, zp, thp1]
    #dyp/dt = ypp = [xpp, zpp, thpp1]
    x = S[4]
    z = S[5]
    th1 = S[6]
    xp = S[7]
    zp = S[8]
    thp1 = S[9]
    th0 = S[10]
    thp0 = S[11]
    #nota: in tesi u tilde è il vettore che qua è u barrato e viceversa. Qua viene seguita la notazione dell'articolo
    utilde = array([[ut], [tau]])
    utildep = array([[utp], [taup]])
    y = array([[x], [z], [th1]])
    yp = array([[xp], [zp], [thp1]])

    #matrici di controllo K
    K3 = np.diag([K[0], K[1], K[2]])
    K2 = np.diag([K[3], K[4], K[5]])
    K1 = np.diag([K[6], K[7], K[8]])
    K0 = np.diag([K[9], K[10], K[11]])
    
    #traiettorie
    yd = array([traj[0], traj[1], traj[2]], dtype = 'float64') #posizione
    ypd = array([traj[3], traj[4], traj[5]], dtype = 'float64') #velocità
    y2pd = array([traj[6], traj[7], traj[8]], dtype = 'float64')  #acceleraione
    y3pd = array([traj[9], traj[10], traj[11]], dtype = 'float64') #jerk
    y4pd = array([traj[12], traj[13], traj[14]], dtype = 'float64') 
    

    R1 = array([[cos(th1), sin(th1)], [-sin(th1), cos(th1)]]) #matrice rotazione
    #derivata R rispetto a th
    dR_dth1 = array([[-sin(th1), cos(th1)], [-cos(th1), -sin(th1)]])
    dgx = dg*sin(th0)
    dgy = dg*cos(th0)
    dgv = array([[dgx], [dgy]]) #vettore distanza centro di rotazione quadrotor con punto di applicazione braccio
    d11 = array([[d1*sin(th1)], [d1*cos(th1)]]) #vettore del braccio
    #m01 = mlink*(dR/dtheta)*d11
    m01 = (np.matmul(dR_dth1, d11))*mlink
    gv = array([[.0], [-m0*g], [.0], [-g*m01[1]]], dtype = 'float64')

    G = array([[-sin(th0), 0, 0], [-cos(th0), 0, 0], [dgx, 1, -1],[0, 0, 1] ], dtype = 'float64')
    M = array([[m0, 0, 0, m01[0]], [0, m0, 0, m01[1]], [0, 0, J0, 0], [m01[0], m01[1], 0, Jlink]], dtype='float64')

    #introduco questi tre termini per compattezza e rendere meno confusi e lunghi i vettori
    pl = sin(th1)*cos(th1)
    fg = sin(th1)**2-cos(th1)**2
    km = mlink*d1
    #derivata di m01 rispetto a th1, necessaria per il vettore c
    dm01_dth1 = array([[-4*pl*km],[2*km*fg]])
    cv = array([[dm01_dth1[0]*thp1**2], [dm01_dth1[1]*thp1**2], [.0], [.0]], dtype = 'float64')
    

    cvb = np.delete(cv, 2, 0) #ctilde (col terzo elemento rimosso)
    gvb = np.delete(gv, 2, 0) #gtilde (col terzo elemento rimosso)
    #definisco anche Mtilde e G tilde necessarie per calclolare y2p
    Mb = np.array([[m0, 0, m01[0]], [0, m0, m01[1]], [m01[0], m01[1], Jlink]], dtype='float64')
    Gt = array([[-sin(th0), 0], [-cos(th0), 0], [0, 1]], dtype = 'float64')
    #calcolo y2p dalla quale ricavo thpp1 da utilizzare per calcolare cvp, gvp e in generale y2p
    y2p = inv(Mb)@(Gt@utilde - cvb - gvb)

    xpp = y2p[0]
    zpp = y2p[1]
    thpp1 = y2p[2]
    
    #derivata prima e seconda di m01 rispetto al tempo
    dm01_dt = array([[-4*thp1*pl*km], [2*thp1*fg*km]])
    dm01_dt2 = array([[km*(-4*thpp1*pl+thp1**2*fg)], [km*(2*thpp1*fg+2*thp1**2*pl)]],dtype = 'float64')
    

    #tutto ciò che è necessario per calcolare y3p
    v = array([[-sin(th0)], [-cos(th0)]])
    vtildep = array([[-cos(th0)*thp0], [sin(th0)*thp0], [0]], dtype = 'float64')
    cvp = array([[km*(-4*thpp1*thp1*pl+2*fg*thp1**3)], [(4*thp1*th1*fg+2*thp1**3*pl)*km], [0]], dtype = 'float64')
    gvp = array([[0], [0], [-dm01_dt[1]*g]],dtype = 'float64')
    Mbp = array([[0, 0, dm01_dt[0]], [0, 0, dm01_dt[1]], [dm01_dt[0], dm01_dt[1], 0]],dtype = 'float64')
    #calcolo y3p dal quale ricavo th3p1 per calcolare cvpp e gvpp che usiamo per trovare ub
    y3p = inv(Mb)@(Gt@utildep+vtildep*ut-Mbp@y2p - cvp - gvp)

    x3p = y3p[0]
    z3p = y3p[1]
    th3p1 = y3p[2]

    cvpp = array([[km*(-8*th3p1*thp1*pl-8*thpp1**2*pl+12*thpp1*2*thp1*fg-16*thp1**4*pl)], [km*(4*th3p1*thp1*fg+16*thpp1**2*fg+24*thpp1*thp1**2*pl-8*thp1**4*fg)], [0]], dtype = 'float64')
    gvpp = array([[0], [0], [-dm01_dt2[1]*g]],dtype = 'float64')
    
    #derivata seconda della matrice M 
    Mbpp = array([[0, 0, dm01_dt2[0]], [0, 0, dm01_dt2[1]], [dm01_dt2[0], dm01_dt2[1] , 0]], dtype = 'float64')
    
    h = array([[-cos(th0)], [sin(th0)]])
    
    Gb = array([[-sin(th0), -cos(th0)*ut/J0, 0], [-cos(th0),  sin(th0)*ut/J0, 0], [0, 0, 1]],dtype = 'float64')
    
    #vpp = array([[-cos(th0)*thpp0+sin(th0)*thp0**2], [-sin(th0)*thpp0-cos(th0)*thp0**2]],dtype = 'float64')

    #errori 
    e3p = y3pd.reshape(3,1)-y3p
    e2p = y2pd.reshape(3,1)-y2p
    ep = ypd.reshape(3,1)-yp
    e = yd.reshape(3,1)-y
    #termine controllato y4p_r
    y4p = y4pd.reshape(3,1) + matmul(K3,e3p) + matmul(K2,e2p)+matmul(K1,ep)+ matmul(K0,e)
    #termine lambda, che in lambd viene esteso nella terza dimensione per essere sommato vettorialmente
    lam = ut/J0*h*(dgx*ut- tau)- v*ut*thp0**2
    lambd = np.array([[lam[0]], [lam[1]], [0]], dtype = 'float64')
    #print(matmul(K3,(y3pd-y3p)))
    #ub = [uppt, ur, taupp]T
    ub = ((np.linalg.inv(Gb))@(Mb@y4p+2*(Mbp@(y3p))+ Mbpp@(y2p)+cvpp+gvpp-2*(vtildep*utp)-lambd)).reshape(3)
    
    #adesso ho trovato la utpp, la ur e taupp posso usare ur e i valori precedentemente integrati per trovare thpp0
    thpp0 = (dgx*ut-tau+ub[1])/J0
    
    # Sd è il vettore stato che restituisco e integro sono le uscite di controllo differenziate e yp, ypp che integrate mi danno y e yp
    #infine c'e thp0 e thpp0
    Sd = zeros(12)
    utd = np.array([[utp], [taup]]).reshape(2)
    ubt = np.array([ub[0], ub[2]]).reshape(2)
    yppt = y2p.reshape(3)
    ypt = yp.reshape(3)
    Sd = np.hstack((utd, ubt, ypt, yppt, thp0, thpp0))
    
    return Sd


#traj =array([trayd, traypd, tray2pd, tray3pd, tray4pd])
#ut deve essere divero da zero sennò mi dà errore perchè Gb non è invertibile per ut = 0
y0 = zeros((12))
y0 = [m0*g, 0, 0, 0, 0, 0,5, 0, 0, 0, 0, 0, 0]
#y0 = ut, tau, utp, taup, x, z, th1, xp, zp, thp1, th0, thp0

#coefficienti della polinomiale di traiettoria 
K = array([107, 95, 300, 149, 243, 3870, 137, 174, 5890, 12, 36, 2400])
#K = Kx3p, Kz3p, Kt3p, Kx2p, Kz2p, Kt2p, Kxp, Kzp, Ktp, Kx, Kz, Kth


#inizializzazione dei vettori delle variabili
ut = zeros(nstep)
tau = zeros(nstep)
x = zeros(nstep)
z = zeros(nstep)
th1 = zeros(nstep)
thp1 = zeros(nstep)
th0 = zeros(nstep)
thp0 = zeros(nstep)
utp = zeros(nstep)
taup = zeros(nstep)
xp = zeros(nstep)
zp = zeros(nstep)
traj = zeros(15)
z2p = zeros(nstep)
z3p = zeros(nstep)
x2p = zeros(nstep)
x3p = zeros(nstep)
t12p = zeros(nstep)
t13p = zeros(nstep)
t02p = zeros(nstep)
t03p = zeros(nstep)

for i in range (0, nstep, 1):
    if i >= 1:
        tempo = i*step 
        traj = zeros(15)
        #vettore traiettoria 
        traj = [xd[i], zd[i], t1d[i], xpd[i], zpd[i], t1pd[i], x2pd[i], z2pd[i], t12pd[i], x3pd[i], z3pd[i], t13pd[i], x4pd[i], z4pd[i], t14pd[i]]
        #lo stato iniziale della nuova integrazione è lo stato attuale precedente
        y0 = [ut[i-1], tau[i-1], utp[i-1], taup[i-1], x[i-1], z[i-1], th1[i-1], xp[i-1], zp[i-1], thp1[i-1], th0[i-1], thp0[i-1]]
        sol = sc.integrate.solve_ivp(sisder, t_span = [0, step], y0 = y0, t_eval = [0, step],  args = [K, traj])
        #print(sol)
        ut[i] = sol.y[0,1] #deve essere sempre diverso da zero
        tau[i] = sol.y[1,1]
        utp[i] = sol.y[2,1]
        taup[i] = sol.y[3,1]
        x[i] = sol.y[4,1]
        z[i] = sol.y[5,1]
        th1[i] = sol.y[6,1]
        xp[i] = sol.y[7,1]
        zp[i] = sol.y[8,1]
        thp1[i] = sol.y[9,1]
        th0[i] = sol.y[10,1]
        thp0[i] = sol.y[11,1]
        #vengono introdotti dei disturbi (attriti, vento, rumore nelle misurazuìioni, etc) di valore randomico
        #i valori sono molto piccoli perchè si ricorda che sono variazioni che avvengono in un centesimo di secondo
        #probabilmente è il caso di diminuire la frequenza dei disturbi 
        distlin = 0.0005
        distang = 0.00008
        disx = uniform(distlin, -distlin)
        disz = uniform(distlin, -distlin)
        disth = uniform(distang, -distang)
        z[i] = z[i] + disz
        x[i] = x[i] + disx
        th1[i] = th1[i] + disth
        #vettori delle derivate di ordine maggiore delle variabili, utili per il plot e per verificare il tracciamento della dinamica ad ordini superiori
        #dal momento che il controllo agisce su una dinamica del quarto ordine
        
        z2p[i] = (zp[i]-zp[i-1])/step
        z3p[i] = (z2p[i]-z2p[i-1])/step
        x2p[i] = (xp[i]-xp[i-1])/step
        x3p[i] = (x2p[i]-x2p[i-1])/step
        t12p[i] = (thp1[i]-thp1[i-1])/step
        t13p[i] = (t12p[i]-t12p[i-1])/step
        t02p[i] = (thp0[i]-thp0[i-1])/step
        t03p[i] = (t02p[i]-t02p[i-1])/step
    else:
        
        ut[i] = m0*g
        tau[i] = 0
        utp[i] = 0
        taup[i] = 0
        x[i] = 0
        z[i] = 0.5
        th1[i] = 0
        xp[i] = 0
        zp[i] = 0
        thp1[i] = 0
        th0[i] = 0
        thp0[i] = 0
    


#visualizzazione
pl.figure (1)

pl.subplot(311)
pl.plot(t, z, color = 'blue', label = 'z')
pl.plot(t, zd, color = 'red', linestyle = '--', label = 'ref z')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(312)
pl.plot(t, x, color = 'blue', label = 'x')
pl.plot(t, xd, color = 'red', linestyle = '--', label = 'ref x')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(313)
pl.plot(t, th1, color = 'blue', label = 'th1')
pl.plot(t, t1d, color = 'red', linestyle = '--', label = 'th1 ref')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)


pl.figure (2)

pl.subplot(311)
pl.plot(t, zp, color = 'blue', label = 'zp')
pl.plot(t, zpd, color = 'red', linestyle = '--', label = 'ref zp')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(312)
pl.plot(t, xp, color = 'blue', label = 'xp')
pl.plot(t, xpd, color = 'red', linestyle = '--', label = 'ref xp')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(313)
pl.plot(t, thp1, color = 'blue', label = 'thp1')
pl.plot(t, t1pd, color = 'red', linestyle = '--', label = 'ref t1p')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.figure (3)

pl.subplot(311)
pl.plot(t, x2p, color = 'blue', label = 'x2p')
pl.plot(t, x2pd, color = 'red', linestyle = '--', label = 'ref x2p')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(312)
pl.plot(t, z2p, color = 'blue', label = 'z2p')
pl.plot(t, z2pd, color = 'red', linestyle = '--', label = 'ref z2p')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(313)
pl.plot(t, t12p, color = 'blue', label = 't12p')
pl.plot(t, t12pd, color = 'red', linestyle = '--', label = 'ref t12p')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)


pl.figure (4)
pl.subplot(311)
pl.plot(t, x3p, color = 'blue', label = 'x3p')
pl.plot(t, x3pd, color = 'red', linestyle = '--', label = 'ref x3')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(312)
pl.plot(t, z3p, color = 'blue', label = 'z3p')
pl.plot(t, z3pd, color = 'red', linestyle = '--', label = 'ref z3p')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(313)
pl.plot(t, t13p, color = 'blue', label = 't13p')
pl.plot(t, t13pd, color = 'red', linestyle = '--', label = 'ref t13p')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.figure (5)

pl.subplot(311)
pl.plot(t, ut, color = 'blue', label = 'up')
pl.plot(t, utp, color = 'red', linestyle = '--', label = 'utp')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(312)
pl.plot(t, tau, color = 'blue', label = 'tau')
pl.plot(t, taup, color = 'red', linestyle = '--', label = 'taup')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.subplot(313)
pl.plot(t, th0, color = 'blue', label = 'th0')
pl.plot(t, thp0, color = 'red',  linestyle = '--', label = 'thp0')
pl.legend(fontsize = 'x-small', loc = 4)
pl.grid(visible = True, axis = 'both', color = 'black', linestyle = '-', linewidth = 0.5)
pl.show()


#animazione

fig = pl.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 4), ylim=(0, 3))
ax.set_aspect('equal')
ax.grid()


patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g', rotation_point= 'center'))

line1, = ax.plot([], [], marker = "o", lw=3)
line2, = ax.plot([], [], marker = "o",  lw=3)
giunto, = ax.plot([], [], '-', lw = 2, color = 'red')
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

qcx = 0.3
qcz = 0.15
l = 0.1
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    giunto.set_data([], [])
    time_text.set_text('')
    patch.set_xy((-qcx/2, -qcz/2))
    patch.set_width(qcx)
    patch.set_height(qcz)
    return line1, line2, giunto, time_text, patch


def animate(i):
    xl1 = [x[i]-qcx/2+qcz/2*sin(th0[i]), x[i]-qcx/2-l*cos(th0[i])]
    yl1 = [z[i], z[i]-l*sin(th0[i])]
    line1.set_data(xl1, yl1)
    xl2 = [x[i]+qcx/2, x[i]+l*cos(th0[i])+qcx/2]
    yl2 = [z[i], z[i]+l*sin(th0[i])]
    line2.set_data(xl2, yl2)
    gx = [x[i], x[i]+l*sin(th1[i])]
    gz = [z[i]-qcz/2, z[i]-l*cos(th1[i])-qcz/2]
    giunto.set_data(gx, gz)
    time_text.set_text(time_template % (i*step))
    patch.set_x(x[i]-qcx/2)
    patch.set_y(z[i]-qcz/2)
    patch.set_angle(th0[i]*180/pi)
    return line1, line2, giunto, time_text, patch

ani = man.FuncAnimation(fig, animate, np.arange(1, len(x)),
                              interval=25, blit=True, init_func=init)

pl.show()