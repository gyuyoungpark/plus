#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "reduce.hpp"
#include "system.hpp"



class System;

class CuInterParameter;

class InterParameter {
 public:
   explicit InterParameter(std::shared_ptr<const System> system, real value);

   // TODO: Necessary to explicitly destroy GpuBuffer members?
   ~InterParameter() {};

   GpuBuffer<uint> uniqueRegions() const;
   const GpuBuffer<real>& values() const;
   size_t numberOfRegions() const;

   // TODO: implement get()
   void set(real value);
   void setBetween(uint i, uint j, real value);

   CuInterParameter cu() const;

 private:
    GpuBuffer<uint> uniqueRegions_;
    std::shared_ptr<const System> system_;
    GpuBuffer<real> valuesbuffer_;
    size_t numRegions_; // TODO: cast into (u)int?

};

class CuInterParameter {
 private:
   uint* regPtr_;
   real* valuePtr_;
   size_t numRegions_;

 public:
   explicit CuInterParameter(const InterParameter* p);
   __device__ real valueBetween(uint i, uint j) const;
   __device__ real valueBetween(int3 coo1, int3 coo2) const;
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
  int i = getIdxOnThread(regPtr_, numRegions_, idx1);
  int j = getIdxOnThread(regPtr_, numRegions_, idx2);
  return valuePtr_[getLutIndex(i, j)];
}