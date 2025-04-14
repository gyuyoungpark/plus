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
  if (magnet->msat.assuredZero()) { return true; }
  if (magnet->hostMagnet<Antiferromagnet>())
    return magnet->hostMagnet<Antiferromagnet>()->dmiVector.assuredZero();
  if (magnet->hostMagnet<NCAFM>())
    return 0;
  return 0;
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
  return hField;
}

FM_FieldQuantity homoDmiFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalHomoDmiField, 3, "homogeneous_dmi_field", "T");
}