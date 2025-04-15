#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "local_dmi.hpp"
#include "ncafm.hpp"
#include "parameter.hpp"
#include "world.hpp"

bool homoDmiAssuredZero(const Ferromagnet* magnet) {
  // Functions returns true if magnet is no sublattice
  bool dmiVectorIsZero = true;
  if (magnet->hostMagnet<Antiferromagnet>())
    dmiVectorIsZero = magnet->hostMagnet<Antiferromagnet>()->dmiVector.assuredZero();
  else if (magnet->hostMagnet<NCAFM>())
    dmiVectorIsZero = magnet->hostMagnet<NCAFM>()->dmiVector.assuredZero();
  return magnet->msat.assuredZero() || dmiVectorIsZero;
}

__global__ void k_homoDmiFieldAFM(CuField hField,
                                  const CuField m2Field,
                                  const CuVectorParameter dmiVector,
                                  const CuParameter msat,
                                  const real symmetry_factor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;
 
  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m2 = m2Field.vectorAt(idx);
  real3 D = dmiVector.vectorAt(idx);
  hField.setVectorInCell(idx, symmetry_factor * cross(D, m2) / msat.valueAt(idx));
}

__global__ void k_homoDmiFieldNCAFM(CuField hField,
                                  const CuField m2Field,
                                  const CuField m3Field,
                                  const CuVectorParameter dmiVector,
                                  const CuParameter msat,
                                  const real symmetry_factor) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m2 = m2Field.vectorAt(idx);
  real3 m3 = m3Field.vectorAt(idx);
  real3 D = dmiVector.vectorAt(idx);
  real3 term1 = symmetry_factor * cross(D, m2);
  real3 term2 = - symmetry_factor * cross(D, m3);

  hField.setVectorInCell(idx, (term1 + term2) / msat.valueAt(idx));
}

Field evalHomoDmiField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (homoDmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  int ncells = hField.grid().ncells();
  auto msat = magnet->msat.cu();

  if (magnet->hostMagnet<Antiferromagnet>()) {
    auto host = magnet->hostMagnet<Antiferromagnet>();
    auto D = host->dmiVector.cu();
    auto m2 = host->getOtherSublattice(magnet)->magnetization()->field().cu();
    real symmetry_factor = (magnet == host->sub1()) ? 1.0 : -1.0;
    cudaLaunch(ncells, k_homoDmiFieldAFM, hField.cu(), m2, D, msat, symmetry_factor);
  }
  else if (magnet->hostMagnet<NCAFM>()) {
    auto host = magnet->hostMagnet<NCAFM>();
    auto D = host->dmiVector.cu();
    auto sub2 = magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[0];
    auto sub3 = magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[1];
    auto m2 = sub2->magnetization()->field().cu();
    auto m3 = sub3->magnetization()->field().cu();
    real symmetry_factor = (magnet == host->sub2()) ? -1.0 : 1.0;
    cudaLaunch(ncells, k_homoDmiFieldNCAFM, hField.cu(), m2, m3, D, msat, symmetry_factor);
  }
  return hField;
}

Field evalHomoDmiEnergyDensity(const Ferromagnet* magnet) {
  if (homoDmiAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);

  return evalEnergyDensity(magnet, evalHomoDmiField(magnet), 0.5);
}

real evalHomoDmiEnergy(const Ferromagnet* magnet) {
  if (homoDmiAssuredZero(magnet))
    return 0;

  real edens = homoDmiEnergyDensityQuantity(magnet).average()[0];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity homoDmiFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalHomoDmiField, 3, "homogeneous_dmi_field", "T");
}

FM_FieldQuantity homoDmiEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalHomoDmiEnergyDensity, 1,
                          "homogeneous_dmi_emergy_density", "J/m3");
}

FM_ScalarQuantity homoDmiEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalHomoDmiEnergy, "homogeneous_dmi_energy", "J");
}
