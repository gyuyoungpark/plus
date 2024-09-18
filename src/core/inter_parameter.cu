

#include "cudalaunch.hpp"
#include "inter_parameter.hpp"
#include "reduce.hpp"

#include <iostream>


InterParameter::InterParameter(std::shared_ptr<const System> system)
    : system_(system),
      regions_(system->regions()),
      indexMap(system->indexMap),
      uniqueRegions_(GpuBuffer<uint>(system->uniqueRegions)) {
        size_t N = system->uniqueRegions.size();
        valuesbuffer_ = GpuBuffer<real>(N * (N + 1) / 2);
        numRegions_ = N;
      }

InterParameter::~InterParameter() {
    valuesbuffer_.recycle(); // Test this, destroy geometry buffer of FM explicitely, is this necessary???
    regions_.recycle();
}

GpuBuffer<uint> InterParameter::regions() const {
    return regions_;
}

GpuBuffer<uint> InterParameter::uniqueRegions() const {
    return uniqueRegions_;
}

GpuBuffer<real> InterParameter::values() const {
    return valuesbuffer_;
}

size_t InterParameter::numberOfRegions() const {
    return numRegions_;
}

void InterParameter::checkIdxInRegions(uint idx) const {
    if(!idxInRegions(regions_, idx)) {
        throw std::invalid_argument("The region index " + std::to_string(idx)
                                                   + " is not defined.");
    }
}

__global__ void k_setBetween(real* values, int index, real value) {
    values[index] = value;
}

void InterParameter::setBetween(uint idx1, uint idx2, real value) {
        
    // TODO: this function constitutes 5 kernel calls ---> replace by one general k_setBetween
    
    system_->checkIdxInRegions(idx1);
    system_->checkIdxInRegions(idx2);

    // DOES THIS REPLACE INDEXMAP?????
    int i = getIdx(uniqueRegions_.get(), numRegions_, idx1);
    int j = getIdx(uniqueRegions_.get(), numRegions_, idx2);
    
    int index;
    if (i <= j)
        index = j * (j + 1) / 2 + i;
    else
        index = i * (i + 1) / 2 + j;
    
    // TODO: replace by cudaReductionKernel???
    cudaLaunch(1, k_setBetween, valuesbuffer_.get(), index, value);
    return;
}

CuInterParameter InterParameter::cu() const {
    return CuInterParameter(this);
}