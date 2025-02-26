
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

def plot_position_through_time(solloc, t, r):
    if 'particle_trajectories' not in os.listdir(solloc):
        os.mkdir(solloc+'/particle_trajectories')

    for i in range(len(t)):
        plt.clf()
        plt.plot(r[:i, 0], r[:i, 1], color='b', alpha=0.2)
        plt.plot(r[:i, 3], r[:i, 4], color='r', alpha=0.2)
        plt.scatter(r[i, 0], r[i, 1], color='b')
        plt.scatter(r[i, 3], r[i, 4], color='r')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.savefig(solloc+'/particle_trajectories/xy-{:.2f}.png'.format(t[i]))
    return None

def plot_larmor_radii_through_time(solloc, t, v, masses, charges, B0):
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
    plt.savefig(solloc+'/twoparticles-larmorradii.png')

    return None

def plot_xy(solloc, r):
    plt.clf()
    plt.grid(True)
    # 1
    plt.plot(r[:, 0], r[:, 1], color='b', label='Particle 1 (x-y)')
    plt.scatter(r[0, 0], r[0, 1], facecolor='b', edgecolor='none')
    # 2
    plt.plot(r[:, 3], r[:, 4], color='r', label='Particle 2 (x-y)')
    plt.scatter(r[0, 3], r[0, 4], facecolor='r', edgecolor='none')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # plt.title('Motion of Two Charged Particles')
    plt.savefig(solloc+'/twoparticle-xy.png')
    # plt.show()
    return None

def plot_xt(solloc, t, r):
    plt.clf()
    plt.plot(t, r[:, 0], color='b', label='Particle 1 (x)')
    plt.plot(t, r[:, 3], color='r', label='Particle 2 (x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.title('x-position of Two Charged Particles')
    plt.grid(True)
    plt.savefig(solloc+'/twoparticle-xt.png')
    # plt.show()
    return None

def plot_yt(solloc, t, r):
    plt.clf()
    plt.plot(t, r[:, 1], color='b', label='Particle 1 (y)')
    plt.plot(t, r[:, 4], color='r', label='Particle 2 (y)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title('y-position of Two Charged Particles')
    plt.grid(True)
    plt.savefig(solloc+'/twoparticle-yt.png')
    # plt.show()
    return None

def plot_zt(solloc, t, r):
    plt.clf()
    plt.plot(t, r[:, 2], color='b', label='Particle 1 (z)')
    plt.plot(t, r[:, 5], color='r', label='Particle 2 (z)')
    plt.xlabel('t')
    plt.ylabel('z')
    plt.legend()
    plt.grid(True)
    plt.savefig(solloc+'/twoparticle-tz.png')
    return None

def plot_phase_space(solloc, r, v):
    fig,ax=plt.subplots(figsize=(6,10),nrows=3,layout='constrained')
    # x
    ax[0].plot(r[:,0], v[:,0], color='b') # 1
    ax[0].plot(r[:,3], v[:,3], color='r') # 2
    ax[0].scatter(r[0,0], v[0,0], facecolor='b',edgecolor='none') # 1
    ax[0].scatter(r[0,3], v[0,3], facecolor='r',edgecolor='none') # 2
    ax[0].set_title('x-vx')
    # y
    ax[1].plot(r[:,1], v[:,1], color='b') # 1
    ax[1].plot(r[:,4], v[:,4], color='r') # 2
    ax[1].scatter(r[0,1], v[0,1], facecolor='b',edgecolor='none') # 1
    ax[1].scatter(r[0,4], v[0,4], facecolor='r',edgecolor='none') # 2
    ax[1].set_title('y-vy')
    # z
    ax[2].plot(r[:,2], v[:,2], color='b') # 1
    ax[2].plot(r[:,5], v[:,5], color='r') # 2
    ax[2].scatter(r[0,2], v[0,2], facecolor='b',edgecolor='none') # 1
    ax[2].scatter(r[0,5], v[0,5], facecolor='r',edgecolor='none') # 2
    ax[2].set_title('z-vz')
    fig.savefig(solloc+'/phase_space.png')
    fig.clf()
    # plt.show()
    return None

def plot_energy_through_time(solloc, t, v, masses):
    m1, m2 = masses
    plt.clf()
    E1 = 0.5*m1*(v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2)
    E2 = 0.5*m2*(v[:, 3]**2 + v[:, 4]**2 + v[:, 5]**2)
    plt.plot(t, E1, color='b', label='Particle 1 (energy)')
    plt.plot(t, E2, color='r', label='Particle 2 (energy)')
    plt.xlabel('Time')
    plt.ylabel('Energy [J]')
    plt.legend()
    plt.savefig(solloc+'/energy_time.png')
    # plt.show()
    return None

def make_video(solloc):
    filelocation = solloc+'/particle_trajectories/'
    png_files = sort_pngs([i for i in os.listdir(filelocation) if '.png' in i])
    with imageio.get_writer(filelocation+'/movie.gif', mode='I', fps=30) as writer:
        for filename in png_files:
            image = imageio.imread(filelocation+filename)
            writer.append_data(image)
    return None

def plot_all(solloc, t, r, v, masses, charges, B0):
    plot_xy(solloc, r)
    plot_xt(solloc, t, r)
    plot_yt(solloc, t, r)
    plot_zt(solloc, t, r)
    plot_energy_through_time(solloc, t, v, masses)
    plot_phase_space(solloc, r, v)
    plot_larmor_radii_through_time(solloc, t, v, masses, charges, B0)
    # tpp.plot_position_through_time(solloc, t, r)
    # tpp.make_video()

if __name__=='__main__':
    from twoparticleRK import *
