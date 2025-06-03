

import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
#________ 1D System ________##
def ReceptorNeuron(id1, id2, totL, k, lag, amp1=1, amp2=1, signal='pulse', window=10):
    if signal == 'pulse':
        out = {
            id2: amp2 * signal.unit_impulse(totL, k + lag), # type: ignore
            id1: amp1 * signal.unit_impulse(totL, k),  # type: ignore
        }
    elif signal == 'window':
        s1 = np.zeros(totL)
        s1[k: k + window] = 1
        s2 = np.zeros(totL)
        s2[k + lag: k + lag + window] = 1
        out = {
            id1: amp1 * s1,
            id2: amp2 * s2
        }
    return out

def LIF(prm):
    t_eval = np.linspace(prm['t0'], prm['tf'], prm['N'])
    dt = t_eval[1] - t_eval[0]

    V = np.zeros_like(t_eval)
    V[0] = prm['Vinit']

    refractory_counter = 0  # Counts down refractory steps

    for i in range(1, len(t_eval)):
        if refractory_counter > 0:
            V[i] = prm['V0']
            refractory_counter -= 1
            continue

        I1 = prm['I1'][i]
        I2 = prm['I2'][i]
        Vdot = (prm['V0'] - V[i-1] + prm['w1'] * I1 + prm['w2'] * I2) / prm['tau']
        V[i] = V[i-1] + dt * Vdot

        if V[i] >= prm['Vq']:
            V[i] = prm['V0']
            refractory_counter = prm['refr']  # Start refractory period
    return t_eval, V

def LIFNeuron(prm):
    t, V = LIF(prm)
    I_out = np.zeros(prm['N']) 
    I_out[np.where(V>=-56)] = 1*prm['OutA'] 
    return t, I_out, V