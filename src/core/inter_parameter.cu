#include "cudalaunch.hpp"
#include "inter_parameter.hpp"
#include "reduce.hpp"

InterParameter::InterParameter(std::shared_ptr<const System> system, real value)
    : system_(system),
      uniqueRegions_(GpuBuffer<uint>(system->uniqueRegions)) {
        size_t N = system->uniqueRegions.size();
        valuesbuffer_ = GpuBuffer<real>(std::vector<real>(N * (N + 1) / 2, value));
        numRegions_ = N;
      }

GpuBuffer<uint> InterParameter::uniqueRegions() const {
    return uniqueRegions_;
}

const GpuBuffer<real>& InterParameter::values() const {
    return valuesbuffer_;
}

size_t InterParameter::numberOfRegions() const {
    return numRegions_;
}

__global__ void k_set(real* values, real value, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N * (N + 1) / 2)
        return;

    // Set only non-diagonal elements by iterating
    // through cumulative sums
    int j = 0;
    int previous = 0;

    while (idx >= previous + (j + 1)) {
        previous += (j + 1);
        j++;
    }

    int i = idx - previous;
    if (i != j)
        values[idx] = value;
}

void InterParameter::set(real value) {
    int N = numRegions_ * (numRegions_ + 1) / 2;
    cudaLaunch(N, k_set, valuesbuffer_.get(), value, static_cast<int>(numRegions_));
}

__global__ void k_setBetween(real* values, int index, real value) {
    values[index] = value;
}

void InterParameter::setBetween(uint idx1, uint idx2, real value) {
        
    // TODO: this function constitutes 5 kernel calls -> replace by one general k_setBetween?
    
    system_->checkIdxInRegions(idx1);
    system_->checkIdxInRegions(idx2);

    int i = getIdx(uniqueRegions_.get(), numRegions_, idx1);
    int j = getIdx(uniqueRegions_.get(), numRegions_, idx2);

    // Update GPU memory directly
    cudaLaunchReductionKernel(k_setBetween, valuesbuffer_.get(), getLutIndex(i, j), value);
    return;
}

CuInterParameter InterParameter::cu() const {
    return CuInterParameter(this);
}