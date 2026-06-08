#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "magnetorotationfield.hpp"
#include "parameter.hpp"
#include "spinrotationfield.hpp"


bool spinRotationAssuredZero(const Ferromagnet* magnet) {
  // Check if elastodynamics OR prescribed rigid angular velocity is available
  bool hasAngVelSource;
  if (magnet->isSublattice()) {
    hasAngVelSource = magnet->hostMagnet()->enableElastodynamics() ||
                      !magnet->hostMagnet()->rigidAngularVelocity.assuredZero();
  } else {
    hasAngVelSource = magnet->enableElastodynamics() ||
                      !magnet->rigidAngularVelocity.assuredZero();
  }

  return (!hasAngVelSource || !magnet->enableBarnett ||
          magnet->msat.assuredZero());
}


// Barnett effective field: H = omega / gamma  (Tesla)
// Energy: w = -(Ms/gamma) * m . omega
// => H_eff = -1/(mu0 Ms) dw/dm = omega/(mu0 gamma)
// In code units (T): H_code = mu0 * H_eff = omega/gamma
__global__ void k_spinRotationField(CuField hField,
                                    const CuField omegaField,
                                    const CuParameter gamma) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = hField.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  real3 omega = omegaField.vectorAt(idx);
  real g = gamma.valueAt(idx);

  hField.setVectorInCell(idx,
    real3{omega.x / g, omega.y / g, omega.z / g});
}


Field evalSpinRotationField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (spinRotationAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  // Get angular velocity from host magnet if sublattice
  Field omega;
  if (magnet->isSublattice()) {
    omega = evalAngularVelocity(magnet->hostMagnet());
  } else {
    omega = evalAngularVelocity(magnet);
  }

  int ncells = hField.grid().ncells();
  CuParameter gamma = magnet->gamma.cu();

  cudaLaunch(ncells, k_spinRotationField, hField.cu(), omega.cu(),
             gamma);
  return hField;
}


Field evalSpinRotationEnergyDensity(const Ferromagnet* magnet) {
  if (spinRotationAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  // w = -(Ms/gamma) m . omega = -1.0 * Ms * m . (omega/gamma)
  return evalEnergyDensity(magnet, evalSpinRotationField(magnet), 1.0);
}

real evalSpinRotationEnergy(const Ferromagnet* magnet) {
  if (spinRotationAssuredZero(magnet))
    return 0.0;

  real edens = spinRotationEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}


FM_FieldQuantity spinRotationFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalSpinRotationField, 3,
                          "spin_rotation_field", "T");
}

FM_FieldQuantity spinRotationEnergyDensityQuantity(
    const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalSpinRotationEnergyDensity, 1,
                          "spin_rotation_energy_density", "J/m3");
}

FM_ScalarQuantity spinRotationEnergyQuantity(
    const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalSpinRotationEnergy,
                           "spin_rotation_energy", "J");
}
