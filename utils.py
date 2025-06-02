import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal
from tqdm import tqdm
plt.rcParams['text.usetex'] = False

##________ 1D System ________##
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


##________ 2D System ________##
def thetaUpdate(O1, O2, theta_R, delta=np.pi):
    i = 0
    hammd =np.sum(O1 != O2)
    while hammd > 0:
        i+=1
        if np.sum(O1)>0:
            dtheta = theta_R[-1] - delta*np.exp(-i)
        elif np.sum(O2)>0:
            dtheta = theta_R[-1] + delta*np.exp(-i)
        theta_R.append(dtheta)
    return theta_R
    
def PeriodicEmission(tmax, period, dt):
    emission = np.arange(0, tmax, period)
    emsindices = (emission/dt).astype(int)
    S_out = np.zeros(int(tmax/dt))
    S_out[emsindices] = 1
    
    return emission, S_out

def PeriodicEmissionW(time, time_params):
    S_out = np.zeros_like(time)
    emission_starts = np.arange(0, time[-1], time_params['period'])

    for t_emit in emission_starts:
        start_idx = int(t_emit / time_params['dt'])
        end_idx = min(start_idx + time_params['width'], len(S_out))
        S_out[start_idx:end_idx] = 1
        
    emission_times = time[S_out == 1]
    return emission_times, S_out


def SimulateTime(df, emission, init_params):
    theta_vals = np.deg2rad(np.arange(0, 360, init_params['theta_resolution']))  # Angular sweep directions
    for i, t in enumerate(tqdm(df['time'], desc="Time iteration:")):
        for t_emit in emission:
            if t < t_emit:
                continue 
            r_wave = init_params['v'] * (t - t_emit)
            hit1 = False
            hit2 = False
            for theta in theta_vals:
                x = init_params['s_position'][0] + r_wave * np.cos(theta)
                y = init_params['s_position'][1] + r_wave * np.sin(theta)
                wave_point = np.array([x, y])

                if not hit1 and np.linalg.norm(wave_point - init_params['r1_position']) < init_params['tolerance']:
                    df.at[i, 'Rout1'] = init_params['r1_amp']
                    hit1 = True

                if not hit2 and np.linalg.norm(wave_point - init_params['r2_position']) < init_params['tolerance']:
                    df.at[i, 'Rout2'] = init_params['r2_amp']
                    hit2 = True
    return df

def SimulateTime2(df, emission, init_params):
    v = init_params['v']
    s_pos = init_params['s_position']
    r1_pos = init_params['r1_position']
    r2_pos = init_params['r2_position']

    arrival1 = emission + np.linalg.norm(r1_pos - s_pos) / v
    arrival2 = emission + np.linalg.norm(r2_pos - s_pos) / v

    for at in arrival1:
        idx = np.searchsorted(df['time'], at)
        if idx < len(df):
            df.at[idx, 'Rout1'] = init_params['r1_amp']

    for at in arrival2:
        idx = np.searchsorted(df['time'], at)
        if idx < len(df):
            df.at[idx, 'Rout2'] = init_params['r2_amp']

    return df

