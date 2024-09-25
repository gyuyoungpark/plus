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
                           std::string name,
                           std::string unit,
                           int ncomp);
    explicit InterParameter(std::shared_ptr<const System> system, real value);


   // TODO: Necessary to explicitly destroy GpuBuffer members?
   ~InterParameter() {};

   size_t numberOfRegions() const;
   std::string name() const { return name_; }
   std::string unit() const { return unit_; }
   int ncomp() const { return ncomp_; }

   GpuBuffer<uint> uniqueRegions() const;
   const GpuBuffer<real>& values() const;

   // TODO: implement get() (beware of empty GpuBuffers!!!)
   void set(real value);
   void setBetween(uint i, uint j, real value);

   CuInterParameter cu() const;

 private:
    std::string name_;
    std::string unit_;
    int ncomp_;

    GpuBuffer<uint> uniqueRegions_;
    std::shared_ptr<const System> system_;
    GpuBuffer<real> valuesbuffer_;
    size_t numRegions_;
};

class CuInterParameter {
 private:
   uint* regPtr_;
   real* valuePtr_;
   size_t numRegions_;

 public:
   explicit CuInterParameter(const InterParameter* p);
   __device__ real valueBetween(uint i, uint j) const;
};

inline CuInterParameter::CuInterParameter(const InterParameter* p)
   : regPtr_(p->uniqueRegions().get()),
     valuePtr_(p->values().get()),
     numRegions_(p->numberOfRegions()) {}


__device__ __host__ inline int getLutIndex(int i, int j) {
  // Look-up Table index
  if (i > j)
    return i * (i - 1) / 2 + j;
  return j * (j - 1) / 2 + i;
}

__device__ inline real CuInterParameter::valueBetween(uint idx1, uint idx2) const {
  return valuePtr_[getLutIndex(idx1, idx2)];
}