

import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
# #________ 1D System ________##
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

def LIF(prm):
    t_eval = np.arange(0, prm['tmax'], prm['dt'])
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
    I_out = np.zeros_like(t) 
    I_out[np.where(V>=prm['Vq']-2)] = prm['OutA'] 
    return t, I_out, V

# ___________________

# Convert simplified 2D spherical (polar-like) to Cartesian: (r, theta) → (x, z)
def spherical_to_cartesian_2d(r, theta):
    x = r * np.sin(theta)
    z = r * np.cos(theta)
    return np.array([x, z])

# Simulation function
def pulse_response(source_coords, r1_coords, r2_coords,
                    timeprms, wave_speed=1.0):
    dt = timeprms['dt']
    time = np.arange(0, timeprms['tmax'], timeprms['dt'])
    ts_len = len(time)
    
    # Convert to 2D Cartesian
    source_pos = spherical_to_cartesian_2d(*source_coords)
    r1_pos = spherical_to_cartesian_2d(*r1_coords)
    r2_pos = spherical_to_cartesian_2d(*r2_coords)
    
    # Distance and arrival time
    d1 = np.linalg.norm(source_pos - r1_pos)
    d2 = np.linalg.norm(source_pos - r2_pos)
    t1 = timeprms['pulse_time'] + d1 / wave_speed
    t2 = timeprms['pulse_time'] + d2 / wave_speed
    
    # Initialize time series
    source_ts = np.zeros(ts_len)
    r1_ts = np.zeros(ts_len)
    r2_ts = np.zeros(ts_len)
    
    # Mark pulse in time series
    def mark_pulse(ts, start_time, duration):
        idx_start = int(start_time / dt)
        idx_end = int((start_time + duration) / dt)
        ts[idx_start:idx_end] = 1.0

    # Trigger pulses
    mark_pulse(source_ts, timeprms['pulse_time'], timeprms['window'])
    mark_pulse(r1_ts, t1, timeprms['window'])
    mark_pulse(r2_ts, t2, timeprms['window'])

    return time, source_ts, r1_ts, r2_ts


def polarscatter(source,iteration, points, title="Self Alignment"):
    assert len(points) == 3, "Input must be a list of exactly 3 (r, θ) tuples"

    r_values = [r for r, theta in points]
    theta_values = [theta for r, theta in points]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Plot first two points in yellow
    ax.scatter(theta_values[0:2], r_values[0:2], color='yellow',alpha=.5, s=500, label="Receptors", edgecolor='black')
    
    # Plot third point in red
    ax.scatter(theta_values[2], r_values[2], color='red', s=60, label="Source", edgecolor='black')

    # Annotate all points
    # for r, theta in points:
        # ax.annotate(f"({r:.1f}, {theta:.2f})", 
        #             (theta, r), 
        #             textcoords="offset points", 
        #             xytext=(5, 5), 
        #             ha='left', fontsize=8)

    # Draw a red line between the first two points
    ax.plot(theta_values[:2], r_values[:2], color='black', linewidth=1)

    ax.set_title(title)
    ax.grid(True)

    # Remove radial (r) ticks
    ax.set_yticklabels([])

    plt.savefig(f'{source:.2f}_{iteration}_polar.png', dpi = 450)
