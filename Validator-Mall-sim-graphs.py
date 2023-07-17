import pickle
from RotationForest import *
# Rotate = pickle.load(open("models/rfc-Mall", 'rb'))
Rotate = pickle.load(open("models/rfc-M12", 'rb'))

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import DOP853
from scipy.signal import resample, decimate

Kt = 3.13e-3
Kd = 7.5e-3
g = 9.81
m = 2
Jr = 0
b = 2.98e-5
d = 1.14e-7
l = 1
Ixx = 5e-3
Iyy = 5e-3
Izz = 1.3e-2

global MMM
global MM
MMM = []

choices = np.diag([1,1,1,1,1,1])

Bx = np.array([
[1.,     1.,    1.,     1.,     1.,    1.,   ],
[-1.,     1.,     0.5 ,  -0.5,   -0.5,    0.5  ],
[ 0.,     0.,    -0.866,  0.866, -0.866,  0.866],
[-1.,     1.,    -1.,     1.,     1.,    -1.   ],
])

# Uncomment for CA
Bx = np.array([
[1.,     1.,    1.,     1.,     1.,    1.,   ],
[-1.,     1.,     0.5 ,  -0.5,   -0.5,    0.5  ],
[ 0.,     0.,    -0.866,  0.866, -0.866,  0.866],
[-1.,     1.,    -1.,     1.,     1.,    -1.   ],
])

# Uncomment for CA F
BxF = np.array([
[0.,     0.,    0.,     0.,     0.,    0.,   ],
[-1.,     1.,     0.5 ,  -0.5,   -0.5,    0.5  ],
[ 0.,     0.,    -0.866,  0.866, -0.866,  0.866],
[0,     0,    0,     0,     0,    0   ],
])

# Bx = Bx*choices
Bx = np.dot(Bx, choices)
print(Bx)

M = Bx
M_inv = np.round(np.linalg.pinv(M), 3)

# For theta and phi (Good Values)
Kp =   1
Ki = 0.01
Kder = 2


Kp_phi = 1
Kp_theta = 1
Kp_psi =   0.4
Kp_th = -8


Kp_phi_n1 = 2 * Kp_phi
Kp_theta_n1 = Kp_theta
Kp_psi_n1 =   2 * Kp_psi
Kp_th_n1 =   (5/6) * Kp_th

#  Comment for normal control
#  Uncomment for f control on M1
#  Kp_phi = Kp_phi_n1
#  Kp_theta = Kp_theta_n1
#  Kp_psi = Kp_psi_n1
#  Kp_th = Kp_th_n1

Ki_psi = 0.0
Kder_psi = 0

Ki_th = 0.0
Kder_th = -200

Phi_des = 0
Tht_des = 0
Psi_des = 0
Z_des = -10

Phi_err = [0]
Tht_err = [0]
Psi_err = [0]
Th_err = [0]

Phi_dif = [0]
Tht_dif = [0]
Psi_dif = [0]
Z_dif = [0]


stINIT = [ 0, 0 ,-10 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0, m*g]
# TODO
# options = odeset('RelTol',1e-3,'AbsTol',1e-5)
#stIN = [x, y, z, phi, theta, psi, u, v, w, p, q, r]
# [t, states] = ode78(@(t,stINIT) hex_dynamics(t,stINIT),tspan,stINIT,options)
stMain = []
tMain = []
i = 1


frequency = 50 #Hertz
tspan = [0, 1/frequency]
timePeriod = 10 #s
# TODO
# sg = @(x) 1/(1 + exp(1)^(-x))

mouts = []

# TODO
def hex_dynamics(t,states):
    Kt = 3.13e-3
    Kd = 7.5e-3
    g = 9.81
    m = 2
    Jr = 0
    b = 2.98e-5
    d = 1.14e-7
    l = 1
    Ixx = 5e-3
    Iyy = 5e-3
    Izz = 1.3e-2

    x = states[0]
    y = states[1]
    z = states[2]
    phi = states[3]
    theta = states[4]
    psi = states[5]
    u = states[6]
    v = states[7]
    w = states[8]
    p = states[9]
    q = states[10]
    r = states[11]
    M_phi = states[12]
    M_theta = states[13]
    M_psi = states[14]
    Tt = states[15]
    # Tt = Tt/(np.cos(theta)*np.cos(phi))
    Taud = [Tt,M_phi,M_theta,M_psi]
    # Taud = np.transpose(Taud)
    # global M M_inv MMM choices
    # MotOut = M_inv*Taud
    MotOut = np.matmul(M_inv, Taud)
    # print(f"Taud: {Taud}")
    # MotOut = min(max(MotOut,0),1000^2)
    # MotOut = min(max(MotOut,0),36)
    MotOut = np.clip(MotOut, 0, 36)
    # print(f"Taud: {MotOut}")
    # Adding fault
    MotOut = np.matmul(MotOut, choices)
    # MMM = [MMM, np.sqrt(MotOut)]
    MMM.append(np.sqrt(MotOut))
    # MM = M*MotOut
    MM = np.matmul(M, MotOut)
    # print(f"MM: {MM}")

    
    M_phi = MM[1]
    M_theta = MM[2]
    M_psi = MM[3]
    Tt = MM[0]
    V = [[u],[v],[w]]

    xyz_dot = [
        [np.cos(theta)*np.cos(psi), (np.cos(psi)*np.sin(phi)*np.sin(theta) - np.cos(phi)*np.sin(psi)), (np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta))],
        [np.cos(theta)*np.sin(psi), (np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi)), (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))],
        [(-np.sin(theta)), (np.cos(theta)*np.sin(phi)), (np.cos(theta)*np.cos(phi))],
        ]
    xyz_dot = np.matmul(xyz_dot, V)
    np.round(xyz_dot, 3)
    # print(f"xyz_dot: {xyz_dot}")

    ptp_dot = [
        [1, (np.sin(phi)*np.tan(theta)), (np.cos(phi)*np.tan(theta))],
        [0, np.cos(phi), (-np.sin(phi))],
        [0, (np.sin(phi)*(1/np.cos(theta))), (np.cos(phi)*(1/np.cos(theta)))]]
    ptp_dot = np.matmul(ptp_dot, [p,q,r])

    alpha_R = 0.5

    uvw_dot = np.matmul([
        [-((Kt+np.multiply(Kd, V[0]))/m)[0], r, -q],
        [-r, -((Kt+np.multiply(Kd, V[1]))/m)[0], p],
        [q, -p, -(np.multiply(Kd, V[2])/m)[0]]], [[u],[v],[w]])
    uvw_dot = uvw_dot + [-g*np.sin(theta), +g*np.sin(phi)*np.cos(theta), +(g*np.cos(phi)*np.cos(theta) - Tt/m)] - [alpha_R*u, alpha_R*v, alpha_R*w]
    
    np.round(uvw_dot, 3)
    # print(f"uvw_dot: {uvw_dot}")
    Omega_R = 0
    Omega_R_dot = 0

    p_dot = ((Iyy - Izz)/Ixx)*q*r + (M_phi - Jr*q*Omega_R)/Ixx - alpha_R*p
    q_dot = ((Izz - Ixx)/Iyy)*p*r + (M_theta + Jr*p*Omega_R)/Iyy - alpha_R*q
    r_dot = ((Ixx - Iyy)/Izz)*p*q + (M_psi + - Jr*Omega_R_dot)/Izz - 5*alpha_R*r

    st_dot = np.array([
        xyz_dot[0][0],
        xyz_dot[1][0],
        xyz_dot[2][0],
        ptp_dot[0],
        ptp_dot[1],
        ptp_dot[2],
        uvw_dot[0][0],
        uvw_dot[0][1],
        uvw_dot[0][2],
        p_dot,
        q_dot,
        r_dot,
        0,
        0,
        0,
        0,
    ])
    np.round(st_dot, 4)
    return st_dot

allStates = []
predictions = []

lastp = 0
lastq = 0
lastr = 0

while i < (timePeriod*frequency):
    x = DOP853(hex_dynamics,tspan[0],stINIT,tspan[1])
    x.step()
    states = np.round(x.y, 4)
    allStates.append(states)
    t = tspan[1]

    # XX = states[len(states),10)
    Phi_err = np.append(Phi_err, (Phi_des - states[3]))
    
    # Tau_phi = Kp*Phi_err(len(Phi_err)) + Ki*sum(Phi_err) + Kder*(Phi_dif(end))
    if(len(Phi_dif) > 0):
        Tau_phi = Kp_phi*Phi_err[len(Phi_err)-1] + Ki*sum(Phi_err) + Kder*(Phi_dif[len(Phi_dif)-1])
    else:
        Tau_phi = Kp_phi*Phi_err[len(Phi_err)-1] + Ki*sum(Phi_err)
    Phi_dif = np.diff(Phi_err)

    Tht_err = np.append(Tht_err, (Tht_des - states[4]))
    # Tau_tht = Kp*Tht_err(len(Tht_err)) + Ki*sum(Tht_err) + Kder*(Tht_dif(end))
    if(len(Tht_dif) > 0):
        Tau_tht = Kp_theta*Tht_err[len(Tht_err)-1] + Ki*sum(Tht_err) + Kder*(Tht_dif[len(Tht_dif)-1])
    else:
        Tau_tht = Kp_theta*Tht_err[len(Tht_err)-1] + Ki*sum(Tht_err)
    Tht_dif = np.diff(Tht_err)

    Psi_err = np.append(Psi_err, (Psi_des - states[5]))
    if(len(Psi_dif) > 0):
        Tau_psi = Kp_psi*Psi_err[len(Psi_err)-1] + Ki_psi*sum(Psi_err) + Kder_psi*(Psi_dif[len(Psi_dif)-1])
    else:
        Tau_psi = Kp_psi*Psi_err[len(Psi_err)-1] + Ki_psi*sum(Psi_err)
    Psi_dif = np.diff(Psi_err)

    Th_err = np.append(Th_err, (Z_des - states[2]))
    if(len(Z_dif) > 0):
        Th = Kp_th*Th_err[len(Th_err)-1] + Ki_th*sum(Th_err) + Kder_th*(Z_dif[len(Z_dif)-1]) + m*g
    else:
        Th = Kp_th*Th_err[len(Th_err)-1] + Ki_th*sum(Th_err) + m*g
    Z_dif = np.diff(Th_err)

    # print("Th: ", Th)

    kk = -2

    if t > 2:
        choices = np.diag([0,1,1,1,1,1])

    if t > 2.25:
        choices = np.diag([0,1,1,1,1,1])
        M = np.dot(Bx, choices)
        M_inv = np.round(np.linalg.pinv(M), 3)

    # R, R-1, RDes, P, P-1, PDes, Y, Y-1, YDes
    Xdata = np.asarray([[states[9], lastp, lastp*kk, states[10], lastq, lastq*kk, states[11], lastr, lastr*kk]])
    preds_rotate = Rotate.predict(Xdata)

    predictions.append(preds_rotate[0])
    
    # print(f"XData: {Xdata}")
    lastp = states[9]
    lastq = states[10]
    lastr = states[11]

    # tMain.append(t)
    tMain = np.append(tMain, t)
    # stMain = np.append(stMain, states)
    stMain.append(states)

    stINIT = states
    stINIT[12] = Tau_phi
    stINIT[13] = Tau_tht
    stINIT[14] = Tau_psi
    stINIT[15] = Th
    # print(f"stINIT[15]: {stINIT[15]}"	)
    tspan[0] = tspan[0] + 1/frequency
    tspan[1] = tspan[1] + 1/frequency
    i = i + 1
    
    tMain = np.array(tMain)
    t = tMain

states = np.array(stMain)
## PLOT RESULTS
t = np.array(t)

print(f"pred: {predictions}")

plt.figure()

plt.subplot(2,4,1)
plt.plot(t,states[:,3])
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
plt.ylim([min(states[:,3])-minScale, max(states[:,3])+minScale])
plt.title(r'$\phi$')
plt.legend([r'$\phi$', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')

plt.subplot(2,4,2)
plt.plot(t,states[:,4])
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
plt.ylim([min(states[:,4])-minScale, max(states[:,4])+minScale])
plt.title(r'$\theta$')
plt.legend([r'$\theta$', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')

plt.subplot(2,4,3)
plt.plot(t,states[:,5])
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
plt.ylim([min(states[:,5])-minScale, max(states[:,5])+minScale])
plt.title(r'$\psi$')
plt.legend([r'$\psi$', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')

plt.subplot(2,4,4)
invZ = (-1.*states[:,2])
plt.plot(t,invZ)
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
plt.ylim([min(invZ)-minScale, max(invZ)+minScale])
plt.title('z')
plt.legend(['z', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')

plt.subplot(2,4,5)
plt.plot(t,states[:,9])
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
plt.ylim([min(states[:,9])-minScale, max(states[:,9])+minScale])
plt.title('p')
plt.legend(['p', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')

plt.subplot(2,4,6)
plt.plot(t,states[:,10])
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
plt.ylim([min(states[:,10])-minScale, max(states[:,10])+minScale])
plt.title('q')
plt.legend(['q', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')

plt.subplot(2,4,7)
plt.plot(t,states[:,11])
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
plt.ylim([min(states[:,11])-minScale, max(states[:,11])+minScale])
plt.title('r')
plt.legend(['r', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')

plt.subplot(2,4,8)
invW = (-1.*states[:,8])
plt.plot(t,invW)
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
plt.ylim([min(invW)-minScale, max(invW)+minScale])
plt.title('w')
plt.legend(['w', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')

MMM = resample(MMM, 499)
plt.figure(2)
plt.plot(t, MMM[:,0])
plt.plot(t, MMM[:,1])
plt.plot(t, MMM[:,2])
plt.plot(t, MMM[:,3])
plt.plot(t, MMM[:,4])
plt.plot(t, MMM[:,5])
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
# plt.ylim([min(MMM.any())-minScale, max(MMM.any())+minScale])
plt.title('Motor Outputs')
plt.legend(['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4', 'Motor 5', 'Motor 6', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Motor Output (scaled 0 to 6)')

plt.figure(3)
plt.plot(t,predictions)
plt.axvline(x = 2, color = 'r')
plt.axvline(x = 2.25, color = 'g')
minScale = 1  # Adjust this value according to your desired minimum scale
plt.ylim([min(predictions)-minScale, max(predictions)+minScale])
plt.title('Rotation Forest Prediction')
plt.legend(['Classifier Output', 'Fault Introduced', 'Fault Corrected'])
plt.xlabel('Time (s)')
plt.ylabel('Classifier Prediction (0 to 6)')	
plt.xlim([1.5, 2.5])
plt.ylim([0, 6])


plt.show()