import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
plt.rcParams['text.usetex'] = True


def Source(time, time_params):
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

def polar2cart(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])
    
def ReceptorN(df, emission, init_params): #SimulateTime2()
    df['outR1'] = np.zeros_like(df['time'])
    df['outR2'] = np.zeros_like(df['time'])
    v = init_params['v']
    s_pos = init_params['Scart']
    r1_pos = init_params['R1cart']
    r2_pos = init_params['R2cart']

    arrival1 = emission + np.linalg.norm(r1_pos - s_pos) / v
    arrival2 = emission + np.linalg.norm(r2_pos - s_pos) / v
    
    for at in arrival1:
        idx = np.searchsorted(df['time'], at)
        df.at[idx, 'outR1'] = init_params['r1_amp']
    for at in arrival2:
        idx = np.searchsorted(df['time'], at)
        df.at[idx, 'outR2'] = init_params['r2_amp']
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
            rfcount = neuronprms['rf'] #refractory period
    return time, V

def InterN(df, neuronprms, timeprms):
    t, V = LIF(df,neuronprms , timeprms)
    Iout = np.zeros_like(df['time'])
    Iout[np.where(V>=(neuronprms['Vq']-1))] = neuronprms['outAmp'] # introduce a lag. 
    df['outI'] = Iout
    Vdf = pd.DataFrame({'time':t})
    Vdf['V_I'] = V
    return df, Vdf

def OuterN(df, Vdf, neuronprms, timeprms): #out_id = {O1, O2}
    _, V = LIF(df,neuronprms , timeprms)
    Iout = np.zeros_like(df['time'])
    Iout[np.where(V>=(neuronprms['Vq']-1))] = neuronprms['outAmp'] # introduce a lag. 
    df[f'out{neuronprms['id']}'] = Iout
    Vdf[f'V_{neuronprms['id']}'] = V
    return df, Vdf




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

    # plt.tight_layout()
    # plt.show()
    


def dfPlot2(df):

    # Ensure 'time' column exists
    if 'time' not in df.columns:
        print("DataFrame must contain a 'time' column.")
        return

    signal_cols = [col for col in df.columns if col != 'time']

    # Set dark theme
    plt.style.use('dark_background')

    # Create subplots
    fig, axes = plt.subplots(len(signal_cols), 1, figsize=(10, 2 * len(signal_cols)), sharex=True)

    if len(signal_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, signal_cols):
        ax.plot(df['time'], df[col], linewidth=1.5, color='cyan')
        ax.set_ylabel(col, rotation=90, color='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Time', color='white')

    fig.patch.set_facecolor('dimgray')
    for ax in axes:
        ax.set_facecolor('dimgray')

    plt.tight_layout()
    plt.show()
