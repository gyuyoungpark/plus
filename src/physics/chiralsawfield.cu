#include "chiralsawfield.hpp"

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "world.hpp"


bool chiralSAWFieldAssuredZero(const Ferromagnet* magnet) {
  return !magnet->enableSAW;
}


/**
 * @brief CUDA kernel: Chiral SAW effective field (MEL + MR + Barnett).
 *
 * Computes all three coupling channels in a single kernel launch,
 * ensuring phase consistency between magnetoelastic, magneto-rotation,
 * and Barnett contributions.
 *
 * For a Rayleigh SAW propagating along x:
 *   eps_xx     = eps0 * sin(k*x - w*t + phi)
 *   Omega_y    = (xi*eps0/2) * cos(k*x - w*t + phi)
 *   d(Omega)/dt = (xi*eps0*w/2) * sin(k*x - w*t + phi)
 *
 * The MR sign flips for -k propagation (encoded in sawDirection).
 */
__global__ void k_chiralSAWField(CuField hField,
                                 const CuField mField,
                                 const CuParameter msat,
                                 const CuParameter B1,
                                 const CuParameter Kmr,
                                 const CuVectorParameter anisU,
                                 const real sawOmega,
                                 const real sawK,
                                 const real sawEps0,
                                 const real sawXi,
                                 const real sawPhi,
                                 const real sawTime,
                                 const int sawDir,
                                 const real gammaLL,
                                 const bool enableMEL,
                                 const bool enableBarnett) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = hField.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  // Cell position: center of cell = (index + 0.5) * cellsize
  const int3 coo = grid.index2coord(idx);
  const real3 cs = system.cellsize;
  real coord;
  if (sawDir == 0) {
    coord = (coo.x + real(0.5)) * cs.x;  // x position
  } else {
    coord = (coo.y + real(0.5)) * cs.y;  // y position
  }

  // SAW phase at this cell
  const real phase = sawK * coord - sawOmega * sawTime + sawPhi;
  const real sin_phase = sin(phase);
  const real cos_phase = cos(phase);

  // Material parameters at this cell
  const real Ms = msat.valueAt(idx);
  const real b1 = B1.valueAt(idx);
  const real kmr = Kmr.valueAt(idx);
  const real3 m = mField.vectorAt(idx);
  const real3 uhat = anisU.vectorAt(idx);

  if (Ms == real(0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 H_total = {0, 0, 0};

  // ---- Channel 1: Magnetoelastic (MEL) ----
  // H_mel,i = -(2*B1/Ms) * eps_ii * m_i  (normal strain only)
  // For SAW along x: eps_xx = eps0 * sin(phase), eps_yy = eps_zz = 0
  if (enableMEL && b1 != real(0)) {
    const real eps_xx = sawEps0 * sin_phase;
    if (sawDir == 0) {
      // Propagation along x: only eps_xx is nonzero
      H_total.x += -2 * b1 * eps_xx * m.x / Ms;
    } else {
      // Propagation along y: only eps_yy is nonzero
      H_total.y += -2 * b1 * eps_xx * m.y / Ms;
    }
  }

  // ---- Channel 2: Magneto-rotation (MR) ----
  // Omega_y = sign * (xi * eps0 / 2) * cos(phase)
  // sign = +1 for +k, -1 for -k (chirality)
  // H_mr = (Kmr/Ms) * [(m . (Omega x u)) * u + (m . u) * (Omega x u)]
  if (kmr != real(0)) {
    const real sign = (sawK > 0) ? real(1) : real(-1);
    const real Omega_y = sign * sawXi * sawEps0 * cos_phase / real(2);
    const real3 Omega = {0, Omega_y, 0};

    // cross = Omega x uhat
    const real3 cross = {Omega.y * uhat.z - Omega.z * uhat.y,
                         Omega.z * uhat.x - Omega.x * uhat.z,
                         Omega.x * uhat.y - Omega.y * uhat.x};

    const real a = m.x * cross.x + m.y * cross.y + m.z * cross.z;  // m . (Omega x u)
    const real b = m.x * uhat.x + m.y * uhat.y + m.z * uhat.z;    // m . u
    const real coeff = kmr / Ms;

    H_total.x += coeff * (a * uhat.x + b * cross.x);
    H_total.y += coeff * (a * uhat.y + b * cross.y);
    H_total.z += coeff * (a * uhat.z + b * cross.z);
  }

  // ---- Channel 3: Barnett ----
  // omega_y = (xi * eps0 * w / 2) * sin(phase)
  // H_Barnett = omega / gamma  (along y)
  if (enableBarnett && gammaLL > real(0)) {
    const real omega_y = sawXi * sawEps0 * sawOmega * sin_phase / real(2);
    H_total.y += omega_y / gammaLL;
  }

  hField.setVectorInCell(idx, H_total);
}


Field evalChiralSAWField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (chiralSAWFieldAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  int ncells = hField.grid().ncells();
  CuField mField = magnet->magnetization()->field().cu();
  CuParameter msat = magnet->msat.cu();
  CuParameter B1 = magnet->B1.cu();
  CuParameter Kmr = magnet->Kmr.cu();
  CuVectorParameter anisU = magnet->anisU.cu();

  // Get current simulation time from the world
  real currentTime = magnet->world()->time();

  cudaLaunch(ncells, k_chiralSAWField,
             hField.cu(), mField, msat, B1, Kmr, anisU,
             magnet->sawFrequency,
             magnet->sawWavevector,
             magnet->sawAmplitude,
             magnet->sawEllipticity,
             magnet->sawPhase,
             currentTime,
             magnet->sawDirection,
             magnet->sawGammaLL,
             magnet->sawEnableMEL,
             magnet->sawEnableBarnett);

  return hField;
}


FM_FieldQuantity chiralSAWFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalChiralSAWField, 3,
                          "chiral_saw_field", "T");
}
