#include "ncafm.hpp"

#include <algorithm>
#include <memory>
#include <math.h>
#include <cfloat>
#include <vector>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "mumaxworld.hpp"

NCAFM::NCAFM(std::shared_ptr<System> system_ptr,
             std::string name)
    : Magnet(system_ptr, name),
      ncafmex_cell(system(), 0.0, name + ":ncafmex_cell", "J/m"),
      ncafmex_nn(system(), 0.0, name + ":ncafmex_nn", "J/m"),
      latcon(system(), 0.35e-9, name + ":latcon", "m"),
      dmiTensor(system()),
      interNCAfmExchNN(system(), 0.0, name + ":inter_ncafmex_nn", "J/m"),
      scaleNCAfmExchNN(system(), 1.0, name + ":scale_ncafmex_nn", ""),
      sub1_(Ferromagnet(system_ptr, name + ":sublattice_1", this)),
      sub2_(Ferromagnet(system_ptr, name + ":sublattice_2", this)),
      sub3_(Ferromagnet(system_ptr, name + ":sublattice_3", this)) {}
      
NCAFM::NCAFM(MumaxWorld* world,
             Grid grid,
             std::string name,
             GpuBuffer<bool> geometry,
             GpuBuffer<unsigned int> regions)
    : NCAFM(std::make_shared<System>(world, grid, geometry, regions), name) {}

const Ferromagnet* NCAFM::sub1() const {
  return &sub1_;
}

const Ferromagnet* NCAFM::sub2() const {
  return &sub2_;
}

const Ferromagnet* NCAFM::sub3() const {
  return &sub3_;
}

std::vector<const Ferromagnet*> NCAFM::getOtherSublattices(const Ferromagnet* sub) const {
  if (std::find(sublattices_.begin(), sublattices_.end(), sub) == sublattices_.end())
    throw std::out_of_range("Sublattice not found in NCAFM.");
  std::vector<const Ferromagnet*> result;
  for (const auto* s : sublattices_) {
      if (s != sub)
          result.push_back(s);
  }
  return result;
}

std::vector<const Ferromagnet*> NCAFM::sublattices() const {
  return sublattices_;
}