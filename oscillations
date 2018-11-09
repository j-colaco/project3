# Oscillatory Phenomena

import random                       # importing random
import numpy as np                  # importing numpy
import matplotlib.pyplot as plt     # importing plots
from scipy.integrate import odeint  # importing odeint

L = float(input("Value for Induction in H: "))  # getting argument from user
C = float(input("Value for capacitance in F: "))  # getting argument from user

M = np.array([5, 1, 0.2])           # creating array of m values
omega0 =  1/(np.sqrt(L*C))          # natural angular frequency
resi = (2*M*(np.sqrt(L/C)))         # resonance curves
y0 = [0,0]                          # initial y values

# time interval
t0 = 0                              # initial time
tmax = (30*np.pi)/omega0            # upper boundary for time
tsteps = 1000                       # number of intervals
t = np.linspace(t0,tmax,tsteps)     # time array
# frequency range
f0 = (0.01*omega0)/(2*np.pi)        # initial frequency
fmax = (2*omega0)/(2*np.pi)         # maximum frequency
steps = 100                         # number of intervals
frange = np.linspace(f0, fmax, steps)  # frequency array


def didt1(y, t, R, omega0):         # defining function for di/dt
    q, dq = y                       # let y be the vector q and q'
    return np.array([dq, -(R/L)*dq - (1/(C*L))*q + (0.5*np.cos(omega0*t))/L])  # returning the derivatives

# Blank arrays for the max current
am = []
bm = []
cm = []
# iterating over the frequency range
for x in frange:
    a = odeint(didt1, y0, t, args=(x*2*np.pi, resi[0]))
    b = odeint(didt1, y0, t, args=(x*2*np.pi, resi[1]))
    c = odeint(didt1, y0, t, args=(x*2*np.pi, resi[2]))
    am.append(np.max(a[500:,1]))
    bm.append(np.max(b[500:,1]))
    cm.append(np.max(c[500:,1]))

# Plotting the Resonance behaviour
plt.subplot(221)                   # subplot 1
plt.plot(frange, am)               # plotting the underdamped case
plt.title("Underdamped Case")	   # plot title
plt.xlabel("Frequency (Hz)")	   # x axis label
plt.ylabel("Current (Amps)")	   # y axis label

plt.subplot(222)                   # subplot 2
plt.plot(frange, bm)               # plotting the critical damping
plt.title("Critically Damped")     # plot title
plt.xlabel("Frequency (Hz)")	   # x axis label
plt.ylabel("Current (Amps)")	   # y axis label

plt.subplot(223)                   # subplot 3
plt.plot(frange, cm)               # plotting the overdamped case
plt.title("Overdamped")            # plot title
plt.xlabel("Frequency (Hz)")	   # x axis label
plt.ylabel("Current (Amps)")	   # y axis label
plt.savefig("resonance.pdf")	   # saving figure
