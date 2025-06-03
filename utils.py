import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal
from tqdm import tqdm
import pandas as pd
import warnings
plt.rcParams['text.usetex'] = False

##________ 1D System ________##
# def ReceptorNeuron(id1, id2, totL, k, lag, amp1=1, amp2=1, signal='pulse', window=10):
#     if signal == 'pulse':
#         out = {
#             id2: amp2 * signal.unit_impulse(totL, k + lag), # type: ignore
#             id1: amp1 * signal.unit_impulse(totL, k),  # type: ignore
#         }
#     elif signal == 'window':
#         s1 = np.zeros(totL)
#         s1[k: k + window] = 1
#         s2 = np.zeros(totL)
#         s2[k + lag: k + lag + window] = 1
#         out = {
#             id1: amp1 * s1,
#             id2: amp2 * s2
#         }
#     return out

# def oldLIF(prm):
#     t_eval = np.linspace(prm['t0'], prm['tf'], prm['N'])
#     dt = t_eval[1] - t_eval[0]

#     V = np.zeros_like(t_eval)
#     V[0] = prm['Vinit']

#     refractory_counter = 0  # Counts down refractory steps

#     for i in range(1, len(t_eval)):
#         if refractory_counter > 0:
#             V[i] = prm['V0']
#             refractory_counter -= 1
#             continue

#         I1 = prm['I1'][i]
#         I2 = prm['I2'][i]
#         Vdot = (prm['V0'] - V[i-1] + prm['w1'] * I1 + prm['w2'] * I2) / prm['tau']
#         V[i] = V[i-1] + dt * Vdot

#         if V[i] >= prm['Vq']:
#             V[i] = prm['V0']
#             refractory_counter = prm['refr']  # Start refractory period
#     return t_eval, V

# def LIFNeuron(prm):
#     t, V = LIF(prm)
#     I_out = np.zeros(prm['N']) 
#     I_out[np.where(V>=-56)] = 1*prm['OutA'] 
#     return t, I_out, V


##________ 2D System ________##
def PeriodicEmission(tmax, period, dt):
    emission = np.arange(0, tmax, period)
    emsindices = (emission/dt).astype(int)
    S_out = np.zeros(int(tmax/dt))
    S_out[emsindices] = 1
    
    return emission, S_out

def PeriodicEmissionW(time, time_params):
    S_out = np.zeros_like(time)
    emission_starts = np.linspace(0, time[-1], time_params['pulses'])
    startidx = []
    for t_emit in emission_starts:
        start_idx = int(t_emit / time_params['dt'])
        end_idx = min(start_idx + time_params['width'], len(S_out))
        S_out[start_idx:end_idx] = 1
        startidx.append(start_idx)
    emission_times = time[S_out == 1]
    return emission_times, S_out, startidx

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

                if not hit1 and np.linalg.norm(wave_point - init_params['R1cart']) < init_params['tolerance']:
                    df.at[i, 'outR1'] = init_params['r1_amp']
                    hit1 = True

                if not hit2 and np.linalg.norm(wave_point - init_params['r2_position']) < init_params['tolerance']:
                    df.at[i, 'outR2'] = init_params['r2_amp']
                    hit2 = True
    return df

def ReceptorN(df, emission, init_params): #SimulateTime2()
    v = init_params['v']
    s_pos = init_params['s_position']
    r1_pos = init_params['R1cart']
    r2_pos = init_params['r2_position']

    arrival1 = emission + np.linalg.norm(r1_pos - s_pos) / v
    arrival2 = emission + np.linalg.norm(r2_pos - s_pos) / v

    for at in arrival1:
        idx = np.searchsorted(df['time'], at)
        if idx < len(df):
            df.at[idx, 'outR1'] = init_params['r1_amp']
        else:
            (print('error'))
    for at in arrival2:
        idx = np.searchsorted(df['time'], at)
        if idx < len(df):
            df.at[idx, 'outR2'] = init_params['r2_amp']
        else:
            (print('error'))
    return df

def LIF(df, neuronprms, timeprms):
    time = df['time']
    dt = timeprms['dt']
    V = np.zeros_like(time)
    V[0] = neuronprms['V0']
    rfcount = 0
    
    for i in range(1, len(time)):
        if rfcount > 0 : 
            V[i] = neuronprms['V0']
            rfcount -= 1
            continue
        I1 = df[neuronprms['I1']].iloc[i] 
        I2 = df[neuronprms['I2']].iloc[i]
        dV = (neuronprms['V0'] - V[i-1] + neuronprms['w1'] * I1 + neuronprms['w2'] * I2) / neuronprms['tau']
        V[i] = V[i-1] +dt * dV
        if V[i] >=neuronprms['Vq']:
            V[i] = neuronprms['V0']
            rfcount = neuronprms['rft'] #refractory period
    return time, V

def InterN(df, neuronprms, timeprms):
    t, V = LIF(df,neuronprms , timeprms)
    Iout = np.zeros_like(df['time'])
    Iout[np.where(V>=-56)] = neuronprms['outA'] # introduce a lag. 
    df['outI'] = Iout
    return df, V

def OuterN(df, neuronprms, timeprms): #out_id = {O1, O2}
    t, V = LIF(df,neuronprms , timeprms)
    Iout = np.zeros_like(df['time'])
    Iout[np.where(V>=-56)] = 1*neuronprms['outA'] # introduce a lag. 
    df[f'out{neuronprms['id']}'] = Iout
    return df, V


# def FirstRun(time_params):
    
#     time = np.arange(0, time_params['tmax'], time_params['dt'])    
#     emission, outS, startidx = PeriodicEmissionW(time, time_params)
    
#     initDF = pd.DataFrame({
#         'time': time,
#         'outS': outS,
#     })
#     return initDF, startidx, emission

# def FindSource(newtime, newSout, emission,
#                 time_params, init_params,
#                 interneuron, outneuron1, outneuron2):
#     outR1 = np.zeros_like(newtime)
#     outR2 = np.zeros_like(newtime)
    
#     df=pd.DataFrame({
#         'time':newtime,
#         'outS':newSout,
#         'outR1': outR1,
#         'outR2': outR2
#     })
    
#     df = ReceptorN(df, emission, init_params)
#     df2, VI = InterN(df, interneuron, time_params)

#     df2, VO1 = OuterN(df2, outneuron1, time_params)
#     df2, VO2 = OuterN(df2, outneuron2, time_params)
#     O1 = True if df2['outO1'].sum() != 0 else False
#     O2 = True if df2['outO2'].sum() != 0 else False
    
#     Vdf = pd.DataFrame({
#         'VI' : VI,
#         'VO1': VO1,
#         'VO2': VO2,
#     })
#     return df2, Vdf, [O1, O2]

class PredictionWarning(UserWarning):
    pass

def RposUpdate(initparams):
    R1r = initparams['R1cart'][0]
    R1theta = initparams['R1cart'][1]
    R2r = initparams['R1cart'][0]
    R2theta = initparams['R1cart'][1]
    R1cart = np.array([R1r * np.cos(R1theta), R1r * np.sin(R1theta)])
    R2cart = np.array([R2r * np.cos(R2theta), R2r * np.sin(R2theta)])
    initparams.update({
        'R1cart':R1cart,
        'R2cart':R2cart,
    })
    return initparams
    
def thetaUpdation(location, init_params, log, delta=np.pi/2):
    O1, O2 = location[-1][0], location[-1][1]  # Look at last value
    if O1 and O2:
        print('O1 and O2 are True.')
        # No change, just update positions to keep consistent
        init_params = ReceptorUpdate(init_params)
    elif O1 and not O2:
        init_params['r1_theta'] -= delta * np.exp(-1 * len(location))
        init_params = ReceptorUpdate(init_params)
        log.append('dtheta -')
    elif not O1 and O2:
        init_params['r2_theta'] += delta * np.exp(-1 * len(location))
        init_params = ReceptorUpdate(init_params)
        log.append('dtheta +')
    else:  # not O1 and not O2
        warnings.warn('Failed in prediction', PredictionWarning)
        # Optionally still update or return original
        init_params = ReceptorUpdate(init_params)

    return init_params

# def initparamsUpdate(init_params):
#     # Source in polar coordinates
#     x_s = init_params['r_s'] * np.cos(init_params['theta_s'])
#     y_s = init_params['r_s'] * np.sin(init_params['theta_s'])
#     source = np.array([x_s, y_s])
    
#     # Receptors (fixed distance from origin along a diameter)
#     R1cart = np.array([init_params['d'], 0])  # theta = pi
#     R2cart = np.array([-init_params['d'],0])  # theta = 0

#     R1polar = np.array([ init_params['d'], 0])  # theta = pi
#     R2polar = np.array([ init_params['d'], np.pi])  # theta = 0
#     init_params.update({'s_position': source,
#                     'R1cart': R1cart,
#                     'R2cart': R2cart,
#                     'R1polar': R1polar,
#                     'R2polar': R2polar,
#                     })
#     return init_params

def polar2cart(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])
    

def ReceptorUpdate(initparams):
    R1r = initparams['d']
    R1theta = initparams['r1_theta']
    R2r = initparams['d']
    R2theta = initparams['r2_theta']
    
    R1cart = polar2cart(R1r, R1theta)
    R2cart = polar2cart(R2r, R2theta)
    
    initparams.update({
        'R1cart':R1cart,
        'R2cart':R2cart,
    })
    return initparams
    ##______________________________##

def dfPlot(df):
    # Ensure 'time' column exists
    if 'time' not in df.columns:
        print("DataFrame must contain a 'time' column.")
        return

    signal_cols = [col for col in df.columns if col != 'time']

    # Create subplots
    fig, axes = plt.subplots(len(signal_cols), 1, figsize=(10, 2 * len(signal_cols)), sharex=True)
    if len(signal_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, signal_cols):
        ax.plot(df['time'], df[col], linewidth=1.5, color = 'red')
        ax.set_ylabel(col, rotation=90)
        # ax.set_yticks([])
        ax.grid(True)
        # ax.set_xlim(4,7)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
