#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "reduce.hpp"
#include "system.hpp"



class System;

class CuInterParameter;

class InterParameter {
 public:
  explicit InterParameter(std::shared_ptr<const System> system,
                           real value,
                           std::string name = "",
                           std::string unit = "");
  explicit InterParameter(std::shared_ptr<const System> system, real value);

  ~InterParameter() {};

  std::string name() const { return name_; }
  std::string unit() const { return unit_; }
  int ncomp() const { return 1; }

  const std::vector<real> eval() const { return valuesbuffer_.getData(); }
  const GpuBuffer<real>& values() const;

  void set(real value);
  void setBetween(uint i, uint j, real value);

  CuInterParameter cu() const;

  // TODO: these user-convenience functions should probably move
  const std::vector<uint> uniqueRegions() const { return system_->uniqueRegions; }
  int numberOfRegions() const { return system_->uniqueRegions.size(); }

 private:
  std::string name_;
  std::string unit_;

  std::shared_ptr<const System> system_;
  GpuBuffer<real> valuesbuffer_;
  size_t valuesLimit_;  // N(N-1)/2 with N = maxRegionIdx+1

  friend CuInterParameter;
};

class CuInterParameter {
 private:
    real* valuePtr_;
    uint valuesLimit_;  // TODO: not needed?

 public:
   explicit CuInterParameter(const InterParameter* p);
   __device__ real valueBetween(uint i, uint j) const;
};

inline CuInterParameter::CuInterParameter(const InterParameter* p)
   : valuePtr_(p->values().get()),
     valuesLimit_(p->valuesLimit_) {}


/** Look-up Table index
 * This transforms a bottom triangular matrix row-first with 0-diagonal
 * to a 1D array index, or an upper triangular matrix column-first.
 * No built-in safety checks for the sake of performance.
 */
__device__ __host__ inline int getLutIndex(int i, int j) {
  if (i > j)
    return i * (i - 1) / 2 + j;
  return j * (j - 1) / 2 + i;
}

__device__ inline real CuInterParameter::valueBetween(uint idx1, uint idx2) const {
  return valuePtr_[getLutIndex(idx1, idx2)];
}