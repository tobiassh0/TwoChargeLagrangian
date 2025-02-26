
import my_constants as const
import twoparticleplotting as tpp

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def runge_kutta_2nd_order(f, r0, v0, t_span, dt, masses, charges, B0, k, gamma=0, alpha_1=0, E0=0):
    """
    Performs 2nd-order Runge-Kutta integration to solve ODEs.

    Args:
        f: The function defining the derivatives (accelerations).
        r0: Initial positions (numpy array, shape (6,)).
        v0: Initial velocities (numpy array, shape (6,)).
        t_span: Tuple (t_start, t_end) defining the time interval.
        dt: Time step.
        masses: Tuple (m1, m2) of the masses.
        charges: Tuple (q1, q2) of the charges.
        B0: Magnetic field strength.
        k: Coulomb's constant.

    Returns:
        t: Array of time points.
        r: Array of positions (shape (len(t), 6)).
        v: Array of velocities (shape (len(t), 6)).
    """

    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)

    r = np.zeros((n_steps, 6))
    v = np.zeros((n_steps, 6))

    r[0] = r0
    v[0] = v0

    for i in range(n_steps - 1):
        k1_r, k1_v = f(r[i], v[i], masses, charges, B0, k, gamma, alpha_1, E0, t[i])  # Evaluate f at (r_i, v_i)
        k2_r, k2_v = f(r[i] + dt*k1_r, v[i] + dt*k1_v, masses, charges, B0, k, gamma, alpha_1, E0, t[i+1]) # Evaluate f at (r_i + dt*k1_v, v_i + dt*k1_r)

        r[i+1] = r[i] + (dt/2) * (k1_r + k2_r)
        v[i+1] = v[i] + (dt/2) * (k1_v + k2_v)

    return t, r, v

def forces(r, v, masses, charges, B0, k, gamma, alpha_1, E0, t):
    """
    Calculates the accelerations (derivatives of velocities).

    Args:
        r: Positions (numpy array, shape (6,)).
        v: Velocities (numpy array, shape (6,)).
        masses: Tuple (m1, m2) of the masses.
        charges: Tuple (q1, q2) of the charges.
        B0: Magnetic field strength.
        k: Coulomb's constant.

    Returns:
        drdt: Velocities (same as input v).
        dvdt: Accelerations (numpy array, shape (6,)).
    """
    m1, m2 = masses
    q1, q2 = charges
    x1, y1, z1, x2, y2, z2 = r
    x1dot, y1dot, z1dot, x2dot, y2dot, z2dot = v

    r_vec = np.array([x2 - x1, y2 - y1, z2 - z1])
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:  # Avoid division by zero
        r_mag = 1e-10 # TODO; handle differently

    # force from coulomb potential energy
    coulomb_force = -(k * q1 * q2 * r_vec) / r_mag**3

    # additive force from constant energy [E_0 * alpha_i * exp(gamma * t)] 
    additive_force_1 = 0 #gamma * np.sqrt((E0 * alpha_1 * m1/ 2)) * np.exp(gamma * t)
    additive_force_2 = 0 #gamma * np.sqrt((E0 * (1 - alpha_1) * m2/ 2)) * np.exp(gamma * t)
    dvdt = np.zeros(6)

    # species 1 (ax, ay, az)
    dvdt[0] = (q1 * B0 * y1dot / m1) + coulomb_force[0] / m1 + additive_force_1 / m1
    dvdt[1] = (-q1 * B0 * x1dot / m1) + coulomb_force[1] / m1
    dvdt[2] = coulomb_force[2] / m1
    # species 2 (ax, ay, az)
    dvdt[3] = (q2 * B0 * y2dot / m2) - coulomb_force[0] / m2 + additive_force_2 / m2
    dvdt[4] = (-q2 * B0 * x2dot / m2) - coulomb_force[1] / m2
    dvdt[5] = -coulomb_force[2] / m2

    drdt = v.copy()  # velocities are the time derivatives of positions

    return drdt, dvdt

def save_all_data(masses, charges, B0, t, r, v,\
                    pos_df='positions_DF_',vel_df='velocities_DF_'):
    
    initial_params = '_'.join([str(i) for i in np.concatenate((np.array(masses)/const.me,np.array(charges)/const.qe,[B0]),axis=0)])
    homeloc=os.getcwd()
    if 'run_{}'.format(initial_params) not in os.listdir(homeloc):
        os.mkdir('run_{}'.format(initial_params))
    run_file_loc = homeloc+'/run_{}/'.format(initial_params)
    rows = t
    columns_r = ['x1','y1','z1','x2','y2','z2']
    columns_v = ['vx1','vy1','vz1','vx2','vy2','vz2']
    # Make DataFrames
    dfr = pd.DataFrame(data=r,index=rows,columns=columns_r)
    dfv = pd.DataFrame(data=v,index=rows,columns=columns_v)
    # Save DataFrame
    dfr.to_csv(run_file_loc+pos_df+'.csv')
    dfv.to_csv(run_file_loc+vel_df+'.csv')
    return None

def load_all_data(solloc,initial_params,pos_df='positions_DF_',vel_df='velocities_DF_'):
    dfr = pd.read_csv(solloc+pos_df+'.csv',index_col=0)
    dfv = pd.read_csv(solloc+vel_df+'.csv',index_col=0)
    r = dfr.to_numpy()
    v = dfv.to_numpy()
    print('r shape :',r.shape)
    print('v shape :',v.shape)
    t = dfr.index.to_numpy()
    return t, r, v

# Example usage:
m1 = const.mD  # Mass of particle 1
m2 = const.mT  # "    "  particle 2
q1 = const.qe  # Charge of particle 1
q2 = const.qe # "       " particle 2
B0 = 2.0  # Magnetic field strength [T]
masses = [m1, m2]
charges = [q1, q2]
Temp = 0.001*const.keV_to_K # 1eV [K]
proton_gyro_period = 2*const.PI*const.mp/(const.qe*1.0) # B0 = 1.0

thermal_speed = np.sqrt(2 * const.kb * Temp / const.mp)
proton_gyro_radius = np.sqrt(2 * const.kb * Temp * const.mp) / (const.qe * 1.0) # approximated by v = sqrt(2*E/m) with B0 = 1.0

# Initial positions
(x1, y1, z1) = (0.05, 0.0, 0.001)
(x2, y2, z2) = (-0.05, 0.0, -0.001)
r0 = np.array([x1, y1, z1, x2, y2, z2])

# Initial velocities
(vx1, vy1, vz1) = (-1/proton_gyro_period, 1/proton_gyro_period, 0.0)
(vx2, vy2, vz2) = (1/proton_gyro_period, 1/proton_gyro_period, 0.0)
v0 = np.array([vx1, vy1, vz1, vx2, vy2, vz2])

# Time interval and step
t_span = (0.0, 5*proton_gyro_period)
dt = t_span[-1]/1000

print(thermal_speed, proton_gyro_radius, proton_gyro_period, proton_gyro_radius/dt)

# Run Sim

t, r, v = runge_kutta_2nd_order(forces, r0, v0, t_span, dt, masses, charges, B0, const.k, 
                            gamma=1, alpha_1=1, E0=1)
save_all_data(masses, charges, B0, t, r, v)


# Load Sim

initial_params = '_'.join([str(i) for i in np.concatenate((np.array(masses)/const.me,np.array(charges)/const.qe,[B0]),axis=0)])
t, r, v = load_all_data(os.getcwd()+'/run_{}'.format(initial_params)+'/',initial_params)


# # Plotting Scripts

tpp.plot_xy(r)
tpp.plot_xt(t, r)
tpp.plot_yt(t, r)
tpp.plot_energy_through_time(t, v, masses)
tpp.plot_phase_space(r, v)
# tpp.plot_time_z(t, r)
tpp.plot_larmor_radii_through_time(t, v, masses, charges, B0)
# tpp.plot_position_through_time(t, r, initial_params)
# tpp.make_video(initial_params)
