
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import imageio
import re
import pandas as pd

me = 9.1093829e-31 # Electron mass [kg]
qe = 1.6021766E-19 # Electron charge [C]
mp = 1836.2*me # Proton mass
mD = 3671.5*me # Deuteron mass
mT = 5497.93*me # Triton mass 
kb = 1.3806488e-23 # Coulomb's constant [JK^{-1}] # 
femto_metre = 1e-15 # m
keV_to_K = 11.6e6 # K/keV

def sort_pngs(png_files):
    """Sorts PNG filenames numerically after the "xy-" prefix."""

    def extract_number(filename):
        match = re.search(r"xy-(\d+\.\d+|\d+)", filename)  # Look for numbers after "xy-"
        if match:
            return float(match.group(1))
        return float('inf')  # Files without "xy-" or numbers go last

    return sorted(png_files, key=extract_number)

def runge_kutta_2nd_order(f, r0, v0, t_span, dt, masses, charges, B0, kb, gamma=0, alpha_1=0, E0=0):
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
        kb: Coulomb's constant.

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
        k1_r, k1_v = f(r[i], v[i], masses, charges, B0, kb, gamma, alpha_1, E0, t[i])  # Evaluate f at (r_i, v_i)
        k2_r, k2_v = f(r[i] + dt*k1_v, v[i] + dt*k1_r, masses, charges, B0, kb, gamma, alpha_1, E0, t[i+1]) # Evaluate f at (r_i + dt*k1_v, v_i + dt*k1_r)

        r[i+1] = r[i] + (dt/2) * (k1_r + k2_r)
        v[i+1] = v[i] + (dt/2) * (k1_v + k2_v)

    return t, r, v

def forces(r, v, masses, charges, B0, kb, gamma, alpha_1, E0, t):
    """
    Calculates the accelerations (derivatives of velocities).

    Args:
        r: Positions (numpy array, shape (6,)).
        v: Velocities (numpy array, shape (6,)).
        masses: Tuple (m1, m2) of the masses.
        charges: Tuple (q1, q2) of the charges.
        B0: Magnetic field strength.
        kb: Coulomb's constant.

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

    coulomb_force = -(kb * q1 * q2 * r_vec) / r_mag**3

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

def plot_through_time(t, r, initial_params):
    run_file_loc = os.getcwd()+'/run_{}'.format(initial_params)
    if 'particle_trajectories' not in os.listdir(run_file_loc):
        os.mkdir(run_file_loc+'/particle_trajectories')

    for i in range(len(t)):
        plt.clf()
        plt.plot(r[:i, 0], r[:i, 1], color='b', alpha=0.2)
        plt.plot(r[:i, 3], r[:i, 4], color='r', alpha=0.2)
        plt.scatter(r[i, 0], r[i, 1], color='b')
        plt.scatter(r[i, 3], r[i, 4], color='r')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.savefig(run_file_loc+'/particle_trajectories/xy-{:.2f}.png'.format(t[i]))
    return None

def plot_larmor_radii_through_time(t, v, masses, charges, B0):
    m1, m2 = masses
    q1, q2 = charges
    v1 = v[:, :3]
    v2 = v[:, 3:]

    v_perp1 = np.sqrt(np.sum(v1**2,axis=1))
    v_perp2 = np.sqrt(np.sum(v2**2,axis=1))

    larmor_radii_1 = m1*v_perp1/(q1*B0)
    larmor_radii_2 = m2*v_perp2/(q2*B0)

    plt.clf()
    plt.plot(t, larmor_radii_1, color='b')
    plt.plot(t, larmor_radii_2, color='r')
    plt.xlabel('time')
    plt.ylabel('Larmor radii')
    plt.savefig('twoparticles-larmorradii.png')

    return None

def plot_xy(r):
    plt.clf()
    plt.plot(r[:, 0], r[:, 1], label="Particle 1 (x-y)")
    plt.plot(r[:, 3], r[:, 4], label="Particle 2 (x-y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Motion of Two Charged Particles")
    plt.grid(True)
    plt.savefig("twoparticle-xy.png")
    plt.show()
    return None

def plot_xt(t, r):
    plt.clf()
    plt.plot(t, r[:, 0], label="Particle 1 (x)")
    plt.plot(t, r[:, 3], label="Particle 2 (x)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.title("x-position of Two Charged Particles")
    plt.grid(True)
    plt.savefig("twoparticle-xt.png")
    # plt.show()
    return None

def plot_yt(t, r):
    plt.clf()
    plt.plot(t, r[:, 1], label="Particle 1 (y)")
    plt.plot(t, r[:, 4], label="Particle 2 (y)")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.title("y-position of Two Charged Particles")
    plt.grid(True)
    plt.savefig("twoparticle-yt.png")
    # plt.show()
    return None

def plot_time_z(t, r):
    plt.clf()
    plt.plot(t, r[:, 2], label="Particle 1 (z)")
    plt.plot(t, r[:, 5], label="Particle 2 (z)")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.legend()
    plt.grid(True)
    plt.savefig("twoparticle-tz.png")
    return None

def plot_energy_t(t, v):
    plt.clf()
    
    return None

def make_video(initial_params=''):
    run_file_loc=os.getcwd()+'/run_{}'.format(initial_params)
    filelocation = run_file_loc+'/particle_trajectories/'
    png_files = sort_pngs([i for i in os.listdir(filelocation) if '.png' in i])
    with imageio.get_writer(filelocation+'/movie.gif', mode='I', fps=30) as writer:
        for filename in png_files:
            image = imageio.imread(filelocation+filename)
            writer.append_data(image)
    return None

def save_all_data(r0, v0, masses, charges, B0, t, r, v,\
                    pos_df='positions_DF_',vel_df='velocities_DF_'):
    
    initial_params = '_'.join([str(np.around(i,2)) for i in np.concatenate((masses,charges,[B0],r0,v0),axis=0)])
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
    dfr.to_csv(run_file_loc+pos_df+".csv")
    dfv.to_csv(run_file_loc+vel_df+".csv")
    return None

def load_all_data(solloc,initial_params,pos_df='positions_DF_',vel_df='velocities_DF_'):
    dfr = pd.read_csv(solloc+pos_df+".csv",index_col=0)
    dfv = pd.read_csv(solloc+vel_df+".csv")
    r = dfr.to_numpy()
    v = dfv.to_numpy()
    t = dfr.index.to_numpy()
    return t, r, v

# Example usage:
m1 = mp  # Mass of particle 1
m2 = mp  # "    "  particle 2
q1 = qe  # Charge of particle 1
q2 = qe # "       " particle 2
B0 = 1.0  # Magnetic field strength [T]
Temp = 1*keV_to_K # K
proton_gyro_period = 2*np.pi*mp/(qe*B0)

thermal_speed = np.sqrt(2 * kb * Temp / mp)
proton_gyro_radius = np.sqrt(2 * kb * Temp * mp) / (qe * B0) # approximated by v = sqrt(2*E/m)

# Initial positions
(x1, y1, z1) = (femto_metre, 0.0, 0.0)
(x2, y2, z2) = (-femto_metre, 0.0, 0.0)
r0 = np.array([x1, y1, z1, x2, y2, z2])

# Initial velocities
(vx1, vy1, vz1) = (-0.1*thermal_speed, 0.1*thermal_speed, 0.0)
(vx2, vy2, vz2) = (0.1*thermal_speed, 0.1*thermal_speed, 0.0)
v0 = np.array([vx1, vy1, vz1, vx2, vy2, vz2])

# Time interval and step
t_span = (0.0, 20*proton_gyro_period)
dt = proton_gyro_period/1000

print(thermal_speed, proton_gyro_radius, proton_gyro_period, proton_gyro_radius/dt)

# Run Sim

t, r, v = runge_kutta_2nd_order(forces, r0, v0, t_span, dt, [m1,m2], [q1,q2], B0, kb, 
                            gamma=1, alpha_1=1, E0=1)
save_all_data(r0, v0, [m1,m2], [q1,q2], B0, t, r, v)


# Load Sim

initial_params = '_'.join([str(np.around(i,2)) for i in np.concatenate(([m1,m2,q1,q2,B0],r0,v0),axis=0)])
t, r, v = load_all_data(os.getcwd()+'/run_{}'.format(initial_params)+'/',initial_params)


# # Plotting Scripts

plot_xy(r)
plot_xt(t, r)
plot_yt(t, r)
# plot_time_z(t, r)
# plot_larmor_radii_through_time(t, v, (m1, m2), (q1, q2), B0)
# plot_through_time(t, r, initial_params)
# make_video(initial_params)
