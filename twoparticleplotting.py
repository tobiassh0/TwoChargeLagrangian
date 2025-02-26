
import numpy as np
import matplotlib.pyplot as plt
import imageio
import re

def sort_pngs(png_files):
    """Sorts PNG filenames numerically after the "xy-" prefix."""

    def extract_number(filename):
        match = re.search(r'xy-(\d+\.\d+|\d+)', filename)  # Look for numbers after 'xy-'
        if match:
            return float(match.group(1))
        return float('inf')  # Files without 'xy-' or numbers go last

    return sorted(png_files, key=extract_number)

def plot_position_through_time(t, r, initial_params):
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
    plt.plot(r[:, 0], r[:, 1], color='b', label='Particle 1 (x-y)')
    plt.plot(r[:, 3], r[:, 4], color='r', label='Particle 2 (x-y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # plt.title('Motion of Two Charged Particles')
    plt.grid(True)
    plt.savefig('twoparticle-xy.png')
    # plt.show()
    return None

def plot_xt(t, r):
    plt.clf()
    plt.plot(t, r[:, 0], color='b', label='Particle 1 (x)')
    plt.plot(t, r[:, 3], color='r', label='Particle 2 (x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.title('x-position of Two Charged Particles')
    plt.grid(True)
    plt.savefig('twoparticle-xt.png')
    # plt.show()
    return None

def plot_yt(t, r):
    plt.clf()
    plt.plot(t, r[:, 1], color='b', label='Particle 1 (y)')
    plt.plot(t, r[:, 4], color='r', label='Particle 2 (y)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title('y-position of Two Charged Particles')
    plt.grid(True)
    plt.savefig('twoparticle-yt.png')
    # plt.show()
    return None

def plot_time_z(t, r):
    plt.clf()
    plt.plot(t, r[:, 2], label='Particle 1 (z)')
    plt.plot(t, r[:, 5], label='Particle 2 (z)')
    plt.xlabel('t')
    plt.ylabel('z')
    plt.legend()
    plt.grid(True)
    plt.savefig('twoparticle-tz.png')
    return None

def plot_phase_space(r, v):
    fig,ax=plt.subplots(figsize=(6,10),nrows=3,layout='constrained')
    # x
    ax[0].plot(r[:,0], v[:,0], color='b') # 1
    ax[0].plot(r[:,3], v[:,3], color='r') # 2
    ax[0].set_title('x-vx')
    # y
    ax[1].plot(r[:,1], v[:,1], color='b') # 1
    ax[1].plot(r[:,4], v[:,4], color='r') # 2
    ax[1].set_title('y-vy')
    # z
    ax[2].plot(r[:,2], v[:,2], color='b') # 1
    ax[2].plot(r[:,5], v[:,5], color='r') # 2
    print(r[:,5], v[:,5])
    ax[2].set_title('z-vz')
    plt.show()
    fig.savefig('phase_space.png')
    return None

def plot_energy_through_time(t, v, masses):
    m1, m2 = masses
    plt.clf()
    E1 = 0.5*m1*(v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2)
    E2 = 0.5*m2*(v[:, 3]**2 + v[:, 4]**2 + v[:, 5]**2)
    plt.plot(t, E1, color='b', label='Particle 1 (energy)')
    plt.plot(t, E2, color='r', label='Particle 2 (energy)')
    plt.xlabel('Time')
    plt.ylabel('Energy [J]')
    plt.legend()
    plt.savefig('energy_time.png')
    # plt.show()
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


if __name__=='__main__':
    from twoparticleRK import *
