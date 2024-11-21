import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import Antiferromagnet, Grid, World
from mumaxplus.util.constants import GAMMALL

# Antiferromagnetic spinwave dispersion relation

# Numerical parameters
fmax = 5E13          # maximum frequency (in Hz) of the sinc pulse
T = 2E-12            # simulation time (longer -> better frequency resolution)
dt = 1 / (2 * fmax)  # the sample time
dx = 1E-9            # cellsize
nx = 512             # number of cells

# Material/system parameters
Bz = 0.2           # bias field along the z direction
A = 10E-12         # exchange constant
A_nn = -5E-12
A_c = -400E-12
Ms = 400e3         # saturation magnetization
alpha = 0.005      # damping parameter
gamma = GAMMALL    # gyromagnetic ratio
K = 1e3
mu0 = 1.256637062E-6

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
    sub.ku1 = K
    sub.anisU = (0, 0, 1)

magnet.afmex_nn = A_nn
magnet.afmex_cell = A_c
magnet.latcon = dx


Bt = lambda t: (1e2* np.sinc(2 * fmax * (t - T / 2)), 0, 0)
mask = np.zeros(shape=(1, 1, nx))
# Put signal at the center of the simulation box
mask[:, :, nx // 2 - 1:nx // 2 + 1] = 1
for sub in magnet.sublattices:
    sub.bias_magnetic_field.add_time_term(Bt, mask)

magnet.sub1.magnetization = (0, 0, 1)
magnet.sub2.magnetization = (0, 0, -1)
#magnet.minimize()

nt = 1 + int(T / dt)
timepoints = np.linspace(0, T, nt)
outputquantities = {'m': lambda: magnet.sub1.magnetization.eval()}

# Run solver
output = world.timesolver.solve(timepoints, outputquantities)

# Apply the two dimensional FFT
m = np.array(output['m'], dtype=float)
mx = m[:, 0, 0, 0, :]
mx_fft = np.fft.fft2(mx)
mx_fft = np.fft.fftshift(mx_fft)

plt.figure(figsize=(10, 6))

# x- and y-coordinates of FT cell centers
ks = np.fft.fftshift(np.fft.fftfreq(nx, dx) * 2*np.pi)
fs = np.fft.fftshift(np.fft.fftfreq(nt, dt))
# FT cell widths
dk = ks[1] - ks[0]
df = fs[1] - fs[0]
# image extent of k-values and frequencies, compensated for cell-width
extent = [ks[0] - dk/2, ks[-1] + dk/2, fs[0] - df/2, fs[-1] + df/2]

# Show the intensity plot of the 2D FFT
plt.imshow(np.abs(mx_fft)**2, extent=extent,
           aspect='auto', origin='lower', cmap="inferno")

# Plot the analytical derived dispersion relation
k = np.linspace(-np.pi/dx, np.pi/dx, 250)
mu0Ms = mu0 * Ms
He = -4 * A_c / (dx*dx*mu0Ms)
Ha = 2 * K / mu0Ms
Hint = (2*A - A_nn) / mu0Ms
freq_theory = mu0 * gamma / (2*np.pi) * np.sqrt((2*He + Ha + Hint * k**2) * (Ha + Hint * k**2)) + Bz * gamma/(2*np.pi)
plt.plot(k, freq_theory, '--', lw=1)

plt.axhline(gamma * Bz / (2 * np.pi), c='g', ls='--', lw=1)
plt.xlim([-np.pi/dx, np.pi/dx])
plt.ylim([0, fmax])
plt.vlines([-1/dx, 1/dx], 0, fmax, 'r', '--', lw=1)
plt.ylabel("$f$ (Hz)")
plt.xlabel("$k$ (1/m)")
plt.colorbar()
plt.title('Antiferromagnetic Spinwave Dispersion Relation')
plt.show()
