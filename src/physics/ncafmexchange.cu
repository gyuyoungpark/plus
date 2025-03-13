#include "ncafm.hpp"
#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ncafmexchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "inter_parameter.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"

bool inHomoNCAfmExchangeAssuredZero(const Ferromagnet* magnet) {
  if (!magnet->hostMagnet<NCAFM>()) { return true; }

  return ( magnet->hostMagnet<NCAFM>()->ncafmex_nn.assuredZero() ||
           (magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[0]->msat.assuredZero() &&
            magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[1]->msat.assuredZero()));
}

bool homoNCAfmExchangeAssuredZero(const Ferromagnet* magnet) {
  if (!magnet->hostMagnet<NCAFM>()) { return true; }

  return (magnet->hostMagnet<NCAFM>()->ncafmex_cell.assuredZero() ||
          (magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[0]->msat.assuredZero() &&
           magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[1]->msat.assuredZero()));
}

// NC-AFM exchange at a single site
__global__ void k_NCafmExchangeFieldSite(CuField hField,
                                const CuField m2Field,
                                const CuField m3Field,
                                const CuParameter msat2,
                                const CuParameter msat3,
                                const CuParameter ncafmex_cell,
                                const CuParameter latcon) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!hField.cellInGeometry(idx)) {
      if (hField.cellInGrid(idx))
        hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }
    if (msat2.valueAt(idx) == 0. && msat3.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }

  const real l = latcon.valueAt(idx);
  const real A = ncafmex_cell.valueAt(idx);
  real3 h = 4 * A * m2Field.vectorAt(idx) / (l * l * msat2.valueAt(idx));
  h += 4 * A * m3Field.vectorAt(idx) / (l * l * msat3.valueAt(idx));
  hField.setVectorInCell(idx, h);
  }

// NC-AFM exchange between NN cells
__global__ void k_NCafmExchangeFieldNN(CuField hField,
                                const CuField m1Field,
                                const CuField m2Field,
                                const CuField m3Field,
                                const CuParameter ncafmex_nn,
                                const CuInterParameter interExch,
                                const CuInterParameter scaleExch,
                                const CuParameter msat2,
                                const CuParameter msat3,
                                const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const Grid grid = hField.system.grid;

  if (!grid.cellInGrid(idx))
    return;

  if (msat2.valueAt(idx) == 0 && msat3.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real3 m2 = m2Field.vectorAt(idx);
  const real3 m3 = m3Field.vectorAt(idx);
  const real ann = ncafmex_nn.valueAt(idx);

  // accumulate exchange field in h for cell at idx, divide by msat at the end
  real3 h2{0, 0, 0};
  real3 h3{0, 0, 0};
  
  // NC-AFM exchange in NN cells
#pragma unroll
  for (int3 rel_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 coo_ = mastergrid.wrap(coo + rel_coo);

    if(!hField.cellInGeometry(coo_))
      continue;
    
    const int idx_ = grid.coord2index(coo_);
    real delta = dot(rel_coo, system.cellsize);
    
    // TODO: is this gain in performance (2 ifs instead of
    // 2 kernel functions) worth the ugliness?
    if(msat2.valueAt(idx_) != 0) {
      real inter = 0;
      real scale = 1;
      real Aex;

      real3 m2_ = m2Field.vectorAt(idx_);
      real ann_ = ncafmex_nn.valueAt(idx_);

      unsigned int ridx = system.getRegionIdx(idx);
      unsigned int ridx_ = system.getRegionIdx(idx_);

      if (ridx != ridx_) {
        scale = scaleExch.valueBetween(ridx, ridx_);
        inter = interExch.valueBetween(ridx, ridx_);
      }
      
      Aex = (inter != 0) ? inter : harmonicMean(ann, ann_);
      Aex *= scale;

      h2 += Aex * (m2_ - m2) / (delta * delta);
    }

    if(msat3.valueAt(idx_) != 0) {
      real inter = 0;
      real scale = 1;
      real Aex;

      real3 m3_ = m3Field.vectorAt(idx_);
      real ann_ = ncafmex_nn.valueAt(idx_);

      unsigned int ridx = system.getRegionIdx(idx);
      unsigned int ridx_ = system.getRegionIdx(idx_);

      if (ridx != ridx_) {
        scale = scaleExch.valueBetween(ridx, ridx_);
        inter = interExch.valueBetween(ridx, ridx_);
      }
      
      Aex = (inter != 0) ? inter : harmonicMean(ann, ann_);
      Aex *= scale;

      h3 += Aex * (m3_ - m3) / (delta * delta);
    }
  }
  hField.setVectorInCell(idx, h2 / msat2.valueAt(idx) + h3 / msat3.valueAt(idx));
}

Field evalHomogeneousNCAfmExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  
  if (homoNCAfmExchangeAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  // is an extra (local) variable worth the memory overhead from a double look-up?
  auto sub2 = magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[0];
  auto sub3 = magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[1];
  auto otherMag2 = sub2->magnetization()->field().cu();
  auto otherMag3 = sub3->magnetization()->field().cu();
  auto msat2 = sub2->msat.cu();
  auto msat3 = sub3->msat.cu();

  auto ncafmex_cell = magnet->hostMagnet<NCAFM>()->ncafmex_cell.cu();
  auto latcon = magnet->hostMagnet<NCAFM>()->latcon.cu();

  cudaLaunch(hField.grid().ncells(), k_NCafmExchangeFieldSite, hField.cu(),
            otherMag2, otherMag3, msat2, msat3, ncafmex_cell, latcon);
  return hField;
}

Field evalInHomogeneousNCAfmExchangeField(const Ferromagnet* magnet) {

  Field hField(magnet->system(), 3);
  
  if (inHomoNCAfmExchangeAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  auto sub2 = magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[0];
  auto sub3 = magnet->hostMagnet<NCAFM>()->getOtherSublattices(magnet)[1];
  auto mag = magnet->magnetization()->field().cu();
  auto otherMag2 = sub2->magnetization()->field().cu();
  auto otherMag3 = sub3->magnetization()->field().cu();
  auto msat2 = sub2->msat.cu();
  auto msat3 = sub3->msat.cu();
  auto ncafmex_nn = magnet->hostMagnet<NCAFM>()->ncafmex_nn.cu();
  auto inter = magnet->hostMagnet<NCAFM>()->interNCAfmExchNN.cu();
  auto scale = magnet->hostMagnet<NCAFM>()->scaleNCAfmExchNN.cu();

  cudaLaunch(hField.grid().ncells(), k_NCafmExchangeFieldNN, hField.cu(),
            mag, otherMag2, otherMag3, ncafmex_nn, inter, scale, msat2, msat3,
            magnet->world()->mastergrid());
  return hField;
}

Field evalInHomoNCAfmExchangeEnergyDensity(const Ferromagnet* magnet) {
  if (inHomoNCAfmExchangeAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalInHomogeneousNCAfmExchangeField(magnet), 0.5);
}

Field evalHomoNCAfmExchangeEnergyDensity(const Ferromagnet* magnet) {
  if (homoNCAfmExchangeAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalHomogeneousNCAfmExchangeField(magnet), 0.5);
}

real evalInHomoNCAfmExchangeEnergy(const Ferromagnet* magnet) {
  if (inHomoNCAfmExchangeAssuredZero(magnet))
    return 0;
    
  real edens = inHomoNCAfmExchangeEnergyDensityQuantity(magnet).average()[0];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

real evalHomoNCAfmExchangeEnergy(const Ferromagnet* magnet) {
  if (homoNCAfmExchangeAssuredZero(magnet))
    return 0;
    
  real edens = homoNCAfmExchangeEnergyDensityQuantity(magnet).average()[0];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity inHomoNCAfmExchangeFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalInHomogeneousNCAfmExchangeField, 3,
                          "inhomogeneous_exchange_field", "T");
}

FM_FieldQuantity homoNCAfmExchangeFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalHomogeneousNCAfmExchangeField, 3,
                          "homogeneous_exchange_field", "T");
}

FM_FieldQuantity inHomoNCAfmExchangeEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalInHomoNCAfmExchangeEnergyDensity, 1,
                          "inhomogeneous_exchange_energy_density", "J/m3");
}

FM_FieldQuantity homoNCAfmExchangeEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalHomoNCAfmExchangeEnergyDensity, 1,
                          "homogeneous_exchange_energy_density", "J/m3");
}

FM_ScalarQuantity inHomoNCAfmExchangeEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalInHomoNCAfmExchangeEnergy,
                          "inhomogeneous_exchange_energy", "J");
}

FM_ScalarQuantity homoNCAfmExchangeEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalHomoNCAfmExchangeEnergy,
                          "homogeneous_exchange_energy", "J");
}