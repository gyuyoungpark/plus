#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dmitensor.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "inter_parameter.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include "system.hpp"

class NCAFM : public Magnet {
 public:
  NCAFM(std::shared_ptr<System> system_ptr,
         std::string name);

  NCAFM(MumaxWorld* world,
         Grid grid,
         std::string name,
         GpuBuffer<bool> geometry,
         GpuBuffer<unsigned int> regions);
         
  /** Empty destructor
   *
   * Sublattices are destroyed automatically. They are not pointers.
   */
  ~NCAFM() override {};
  
 std::vector<const Ferromagnet*> sublattices() const;
 const Ferromagnet* sub1() const;
 const Ferromagnet* sub2() const;
 const Ferromagnet* sub3() const;
 std::vector<const Ferromagnet*> getOtherSublattices(const Ferromagnet* sub) const;

 void minimize(real tol = 1e-6, int nsamples = 30);
 void relax(real tol);

 public:
  Parameter ncafmex_cell;
  Parameter ncafmex_nn;
  InterParameter interNCAfmExchNN;
  InterParameter scaleNCAfmExchNN;
  Parameter latcon;
  Ferromagnet sub1_;
  Ferromagnet sub2_;
  Ferromagnet sub3_;

  DmiTensor dmiTensor;
 
 private:
  std::vector<const Ferromagnet*> sublattices_ = {&sub1_, &sub2_, &sub3_};
};