import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import warnings
class PredictionWarning(UserWarning):
    pass

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

def dfPlot(df):
    if 'time' not in df.columns:
        print("DataFrame must contain a 'time' column.")
        return

    signal_cols = [col for col in df.columns if col != 'time']

    fig, axes = plt.subplots(len(signal_cols), 1, figsize=(10, 2 * len(signal_cols)), sharex=True)
    if len(signal_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, signal_cols):
        ax.plot(df['time'], df[col], linewidth=1.5, color = 'red')
        ax.set_ylabel(col, rotation=90)
        # ax.set_yticks([])
        ax.grid(True)
        # ax.set_xlim(2,7.6)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()


def eulerLIF(neurondict, I1, I2, rfcount):
    if rfcount > 0:
        V_O2[i] = neurondict['V0']
        rfcount -= 1
    else:
        I1 = df.at[i, I1]
        I2 = df.at[i, I2]
        dV = (neurondict['V0'] - V_O2[i-1] + neurondict['w1']*I1 + neurondict['w2']*I2) / neurondict['tau']
        V_O2[i] = V_O2[i-1] + time_params['dt'] * dV
        if V_O2[i] >= neurondict['Vq']:
            V_O2[i] = neurondict['V0']
            rfcount = neurondict['rft']
    
    df.at[i, 'outO2'] = neurondict['outA'] if V_O2[i] >= -56 else 0
    