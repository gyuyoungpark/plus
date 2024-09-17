#pragma once

#include <unordered_map>

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "system.hpp"



class System;

class CuInterParameter;

class InterParameter {
 public:
   explicit InterParameter(std::shared_ptr<const System> system);

   ~InterParameter();

   GpuBuffer<uint> regions() const;
   GpuBuffer<real> values() const;
   size_t numberOfRegions() const;

   // TODO: implement get()

   void checkIdxInRegions(uint) const; // CAN THIS FUNCTION BE REMOVED???
   void setBetween(uint i, uint j, real value);

   CuInterParameter cu() const;

 public:
   std::unordered_map<uint, uint> indexMap;

 private:
    GpuBuffer<uint> uniqueRegions_;
    std::shared_ptr<const System> system_;
    GpuBuffer<uint> regions_;
    GpuBuffer<real> valuesbuffer_;
    size_t numRegions_; // TODO: cast into (u)int

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
   : regPtr_(p->regions().get()),
     valuePtr_(p->values().get()),
     numRegions_(p->numberOfRegions()) {}

__device__ inline real CuInterParameter::valueBetween(uint i, uint j) const {
   // TODO: implement this
   return i*j;
}

__device__ inline real CuInterParameter::valueBetween(int3 coo1, int3 coo2) const {
   // TODO: implement this
   return coo1.x*coo2.x;
}