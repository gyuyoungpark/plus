#include "ncafm.hpp"

#include <algorithm>
#include <memory>
#include <math.h>
#include <cfloat>
#include <vector>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "relaxer.hpp"

NCAFM::NCAFM(std::shared_ptr<System> system_ptr,
             std::string name)
    : HostMagnet(system_ptr, name),
      dmiTensor(system()),
      dmiVector(system(), real3{0,0,0}, name + ":dmi_vector", "J/m³"),
      sub1_(Ferromagnet(system_ptr, name + ":sublattice_1", this)),
      sub2_(Ferromagnet(system_ptr, name + ":sublattice_2", this)),
      sub3_(Ferromagnet(system_ptr, name + ":sublattice_3", this)) {
        addSublattice(&sub1_);
        addSublattice(&sub2_);
        addSublattice(&sub3_);
      }
      
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

void NCAFM::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void NCAFM::relax(real tol) {
    std::vector<real> threshold = {
        sub1()->RelaxTorqueThreshold,
        sub2()->RelaxTorqueThreshold,
        sub3()->RelaxTorqueThreshold
    };

    auto positive_min = [](real a, real b) -> real {
        if (a > 0.0 && b > 0.0) return std::min(a, b);
        if (a > 0.0) return a;
        if (b > 0.0) return b;
        return -1.0;
    };

    // If only one sublattice has a user-set threshold, then all
    // sublattices are relaxed using the same threshold.
    // If two thresholds are set, propagate the smallest to the remaining one
    for (int i = 0; i < 3; ++i) {
      if (threshold[i] <= 0.0) {
        real a = threshold[(i + 1) % 3];
        real b = threshold[(i + 2) % 3];
        threshold[i] = positive_min(a, b);
      }
    }
    Relaxer relaxer(this, threshold, tol);
    relaxer.exec();
}