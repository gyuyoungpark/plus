import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mumaxplus import Antiferromagnet, Grid, World
from mumaxplus.util.constants import GAMMALL

# Antiferromagnetic spinwave dispersion relation

# Numerical parameters
fmax = 8E12          # maximum frequency (in Hz) of the sinc pulse
T = 1E-11            # simulation time (longer -> better frequency resolution)
dt = 1 / (2 * fmax)  # the sample time
dx = 1E-10           # cellsize
nx = 5120           # number of cells
k = np.linspace(-np.pi/dx, np.pi/dx, 2000)

# Material/system parameters
Bz = 4           # bias field along the z direction
A = 2.48e-12         # exchange constant
A_nn = 0
A_c = -1E-12
Ms = 566e3         # saturation magnetization
alpha = 0.005      # damping parameter
gamma = GAMMALL    # gyromagnetic ratio
K = 4.64631223e-25 / 1.12152364e-28
mu0 = 1.256637062E-6
latcon = 0.35e-9

# magnetoelastic parameters
rho = 2800  # mass density MnPS3
B2 = -1.2e6
B1 = B2
c11 = 245e9*2
c44 = 75e9*2
c12 = c11 - 2*c44  # assume isotropic

# Create the world
grid_size = (nx, 1, 1)
cell_size = (dx, dx, dx)

world = World(cell_size)
world.bias_magnetic_field = (0, 0, Bz)

# Create a ferromagnet
magnet = Antiferromagnet(world, Grid(size=grid_size))
for sub in magnet.sublattices:
    sub.msat = Ms
    sub.aex = A
    sub.alpha = alpha
    sub.B1 = B1
    sub.B2 = B2
    sub.ku1 = K
    sub.anisU = (0,0,1)

magnet.afmex_nn = A_nn
magnet.afmex_cell = A_c
magnet.latcon = latcon

Bt = lambda t: (1e2* np.sinc(2 * fmax * (t - T / 2)), 0, 0)
mask = np.zeros(shape=(1, 1, nx))
# Put signal at the center of the simulation box
mask[:, :, nx // 2 - 1:nx // 2 + 1] = 1
for sub in magnet.sublattices:
    sub.bias_magnetic_field.add_time_term(Bt, mask)

magnet.sub1.magnetization = (0, 0, 1)
magnet.sub2.magnetization = (0, 0, -1)
#magnet.minimize()

magnet.c11 = c11
magnet.c12 = c12
magnet.c44 = c44
magnet.rho = rho

magnet.enable_elastodynamics = True
# time stepping
world.timesolver.adaptive_timestep = False
world.timesolver.timestep = 1e-15

# parameters to save
nt = 1 + int(T / dt)

m = np.zeros(shape=(nt, 3, 1, 1, nx))
u = np.zeros(shape=(nt, 3, 1, 1, nx))

# run a while and save the data
m[0,...] = magnet.sub1.magnetization.eval()
u[0,...] = magnet.elastic_displacement.eval()
for i in tqdm(range(1, nt)):
    world.timesolver.run(dt)
    m[i,...] = magnet.sub1.magnetization.eval()
    u[i,...] = magnet.elastic_displacement.eval()

# plotting ranges
xmin, xmax = 50e6, 5000e6  # rad/m
ymin, ymax = 50e9, 8000e9  # Hz
extent = [-(2 * np.pi) / (2 * dx) * (nx+1)/nx,
          (2 * np.pi) / (2 * dx) * (nx-1)/nx,
          -1 / (2 * dt) * nt/(nt-1),
          1 / (2 * dt) * nt/(nt-1)]

def plot_FT(FT, name=""):
    maximum = np.max(FT[y_start:y_end,x_start:x_end])
    FT /= maximum
    plt.imshow(np.abs(FT), extent=extent, aspect='auto', origin='lower', cmap="inferno", vmin=0, vmax=0.1)
    plt.title(f"{name} {maximum}")

    # magnon wave
    mu0Ms = mu0 * Ms
    He = -4 * A_c / (latcon*latcon*mu0Ms)
    Ha = 2 * K / mu0Ms
    Hint = (2*A - A_nn) / mu0Ms
    omega_theory = mu0 * gamma * np.sqrt((2*He + Ha + Hint * k**2) * (Ha + Hint * k**2)) + Bz * gamma
    plt.plot(k, omega_theory/(2*np.pi), lw=1)
    # elastic waves
    vt = np.sqrt(c44/rho)
    vl = np.sqrt(c11/rho)
    omega_t = np.abs(vt*k)
    omega_l = np.abs(vl*k)
    plt.plot(k, omega_t/(2*np.pi))
    plt.plot(k, omega_l/(2*np.pi))

    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.colorbar()
    plt.savefig(f"{name}.png")
    plt.close()


# normalize them in the relevant area, so they are visible in the plot
x_start = int((xmin - extent[0]) / (extent[1] - extent[0]) * nx)
x_end = int((xmax - extent[0]) / (extent[1] - extent[0]) * nx)
y_start = int((ymin - extent[2]) / (extent[3] - extent[2]) * nt)
y_end = int((ymax - extent[2]) / (extent[3] - extent[2]) * nt)

# Apply the two dimensional FFT
# Fourier in time and x-direction of displacement and magnetization
u_FT = np.zeros((nt, nx))
m_FT = np.zeros((nt, nx))
param = ["x", "y", "z"]
for i in range(3):
    u_FT += np.abs(np.fft.fftshift(np.fft.fft2(u[:,i,0,0,:])))
    m_FT += np.abs(np.fft.fftshift(np.fft.fft2(m[:,i,0,0,:])))
    plot_FT(np.abs(np.fft.fftshift(np.fft.fft2(u[:,i,0,0,:]))), f"u{param[i]}")
    plot_FT(np.abs(np.fft.fftshift(np.fft.fft2(m[:,i,0,0,:]))), f"m{param[i]}")

u_max = np.max(u_FT[y_start:y_end, x_start:x_end])
m_max = np.max(m_FT[y_start:y_end, x_start:x_end])

FT_tot = u_FT/u_max + m_FT/m_max

plt.figure(figsize=(10, 6))

# Show the intensity plot of the 2D FFT
plt.imshow(np.abs(FT_tot)**2, extent=extent, aspect='auto', origin='lower', cmap="inferno", vmin=0, vmax=2)

# Plot the analytical derived dispersion relation
# elastic waves
vt = np.sqrt(c44/rho)
vl = np.sqrt(c11/rho)
omega_t = np.abs(vt*k)
omega_l = np.abs(vl*k)
plt.plot(k, omega_t/(2*np.pi))
plt.plot(k, omega_l/(2*np.pi))

# magnon wave
mu0Ms = mu0 * Ms
He = -4 * A_c / (latcon*latcon*mu0Ms)
Ha = 2 * K / mu0Ms
Hint = (2*A - A_nn) / mu0Ms
omega_theory = mu0 * gamma * np.sqrt((2*He + Ha + Hint * k**2) * (Ha + Hint * k**2)) + Bz * gamma
plt.plot(k, omega_theory/(2*np.pi), lw=1)

plt.axhline(gamma * Bz / (2 * np.pi), c='g', ls='--', lw=1)
plt.xlim([-np.pi/dx, np.pi/dx])
plt.ylim([0, fmax])
plt.vlines([-1/dx, 1/dx], 0, fmax, 'r', '--', lw=1)
plt.ylabel("$f$ (Hz)")
plt.xlabel("$k$ (1/m)")
plt.colorbar()
plt.xlim((xmin, xmax))
plt.ylim((ymin, ymax))
plt.title('Antiferromagnetic Spinwave Dispersion Relation')
plt.savefig("tot.png")
plt.show()
