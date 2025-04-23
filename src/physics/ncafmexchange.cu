#include "ncafm.hpp"
#include "cudalaunch.hpp"
#include "dmi.hpp" // used for Neumann BC
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

// NCAFM exchange between NN cells
__global__ void k_NCafmExchangeFieldNN(CuField hField,
                                const CuField m1Field,
                                const CuField m2Field,
                                const CuField m3Field,
                                const CuParameter aex,
                                const CuParameter ncafmex_nn,
                                const CuInterParameter interExch,
                                const CuInterParameter scaleExch,
                                const CuParameter msat2,
                                const CuParameter msat3,
                                const Grid mastergrid,
                                const CuDmiTensor dmiTensor,
                                bool openBC) {
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
  const real a = aex.valueAt(idx);
  const real ann = ncafmex_nn.valueAt(idx);

  // If there is no FM-exchange at the boundary, open BC are assumed
  openBC = (a == 0) ? true : openBC;

  // accumulate exchange field in h for cell at idx, divide by msat at the end
  real3 h2{0, 0, 0};
  real3 h3{0, 0, 0};
  
  // NCAFM exchange in NN cells
#pragma unroll
  for (int3 rel_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 coo_ = mastergrid.wrap(coo + rel_coo);

    if(!hField.cellInGeometry(coo_) && openBC)
      continue;
    
    const int idx_ = grid.coord2index(coo_);
    real delta = dot(rel_coo, system.cellsize);
    
    // TODO: is this gain in performance (2 if-scopes instead of
    // 2 kernel functions) worth the ugliness?
    if(msat2.valueAt(idx_) != 0 || !openBC) {
      real3 m2_;
      real ann_;
      int3 normal = rel_coo * rel_coo;

      real inter = 0;
      real scale = 1;
      real Aex;

      if (hField.cellInGeometry(coo_)) {
        real3 m2_ = m2Field.vectorAt(idx_);
        real ann_ = ncafmex_nn.valueAt(idx_);

        unsigned int ridx = system.getRegionIdx(idx);
        unsigned int ridx_ = system.getRegionIdx(idx_);

        if (ridx != ridx_) {
          scale = scaleExch.valueBetween(ridx, ridx_);
          inter = interExch.valueBetween(ridx, ridx_);
        }
      }
      else { // Neumann BC
        real3 Gamma2 = getGamma(dmiTensor, idx, normal, m2);

        real3 d_m1{0, 0, 0};
        int3 other_neighbor_coo = mastergrid.wrap(coo - rel_coo);
        if(hField.cellInGeometry(other_neighbor_coo)){
          // Approximate normal derivative of sister sublattice by taking
          // the bulk derivative closest to the edge.
          real3 m1__ = m1Field.vectorAt(other_neighbor_coo);
          real3 m1 = m1Field.vectorAt(idx);
          d_m1 = (m1 - m1__) / delta;
        }

        m2_ = m2 + (ann * cross(cross(d_m1, m2), m2) + Gamma2) * delta / (2*a);
        ann_ = ann;
      }
      Aex = (inter != 0) ? inter : harmonicMean(ann, ann_);
      Aex *= scale;
      h2 += Aex * (m2_ - m2) / (delta * delta);
    }

    if(msat3.valueAt(idx_) != 0 || !openBC) {
      real3 m3_;
      real ann_;
      int3 normal = rel_coo * rel_coo;

      real inter = 0;
      real scale = 1;
      real Aex;
      if (hField.cellInGeometry(coo_)) {
        real3 m3_ = m3Field.vectorAt(idx_);
        real ann_ = ncafmex_nn.valueAt(idx_);

        unsigned int ridx = system.getRegionIdx(idx);
        unsigned int ridx_ = system.getRegionIdx(idx_);

        if (ridx != ridx_) {
          scale = scaleExch.valueBetween(ridx, ridx_);
          inter = interExch.valueBetween(ridx, ridx_);
        }
      }
      else { // Neumann BC
        real3 Gamma3 = getGamma(dmiTensor, idx, normal, m3);

        real3 d_m1{0, 0, 0};
        int3 other_neighbor_coo = mastergrid.wrap(coo - rel_coo);
        if(hField.cellInGeometry(other_neighbor_coo)){
          // Approximate normal derivative of sister sublattice by taking
          // the bulk derivative closest to the edge.
          real3 m1__ = m1Field.vectorAt(other_neighbor_coo);
          real3 m1 = m1Field.vectorAt(idx);
          d_m1 = (m1 - m1__) / delta;
        }

        m3_ = m3 + (ann * cross(cross(d_m1, m3), m3) + Gamma3) * delta / (2*a);
        ann_ = ann;
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
  auto aex = magnet->aex.cu();
  auto ncafmex_nn = magnet->hostMagnet<NCAFM>()->ncafmex_nn.cu();
  auto BC = magnet->enableOpenBC;
  auto dmiTensor = magnet->dmiTensor.cu();
  auto inter = magnet->hostMagnet<NCAFM>()->interNCAfmExchNN.cu();
  auto scale = magnet->hostMagnet<NCAFM>()->scaleNCAfmExchNN.cu();

  cudaLaunch(hField.grid().ncells(), k_NCafmExchangeFieldNN, hField.cu(),
            mag, otherMag2, otherMag3, aex, ncafmex_nn, inter, scale, msat2, msat3,
            magnet->world()->mastergrid(), dmiTensor, BC);
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

__global__ void k_angle(CuField angleField,
                        const CuField mField1,
                        const CuField mField2,
                        const CuField mField3,
                        const CuParameter ncafmex,
                        const CuParameter msat1,
                        const CuParameter msat2,
                        const CuParameter msat3) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!angleField.cellInGeometry(idx)) {
    if (angleField.cellInGrid(idx))
      angleField.setVectorInCell(idx, real3{0, 0, 0});
      return;
  }

  if (ncafmex.valueAt(idx) == 0) {
    angleField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  bool b1 = (msat1.valueAt(idx) != 0);
  bool b2 = (msat2.valueAt(idx) != 0);
  bool b3 = (msat3.valueAt(idx) != 0);

  real dev12 = acos(dot(mField1.vectorAt(idx) * b1, mField2.vectorAt(idx) * b2))
                    - (120.0 * M_PI / 180.0);
  real dev13 = acos(dot(mField1.vectorAt(idx) * b1, mField3.vectorAt(idx) * b3))
                    - (120.0 * M_PI / 180.0);
  real dev23 = acos(dot(mField2.vectorAt(idx) * b2, mField3.vectorAt(idx) * b3))
                    - (120.0 * M_PI / 180.0);

  angleField.setVectorInCell(idx, real3{dev12, dev13, dev23});
}

Field evalAngleField(const NCAFM* magnet) {
  // Three components for the angles between 1-2, 1-3 and 2-3
  Field angleField(magnet->system(), 3);

  cudaLaunch(angleField.grid().ncells(), k_angle, angleField.cu(),
             magnet->sub1()->magnetization()->field().cu(),
             magnet->sub2()->magnetization()->field().cu(),
             magnet->sub3()->magnetization()->field().cu(),
             magnet->ncafmex_cell.cu(),
             magnet->sub1()->msat.cu(),
             magnet->sub2()->msat.cu(),
             magnet->sub3()->msat.cu());
  return angleField;
}

real evalMaxAngle(const NCAFM* magnet) {
return maxAbsValue(evalAngleField(magnet));
}

NCAFM_FieldQuantity angleFieldQuantity(const NCAFM* magnet) {
return NCAFM_FieldQuantity(magnet, evalAngleField, 3, "angle_field", "rad");
}

NCAFM_ScalarQuantity maxAngle(const NCAFM* magnet) {
return NCAFM_ScalarQuantity(magnet, evalMaxAngle, "max_angle", "rad");
}