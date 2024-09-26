#include "cudalaunch.hpp"
#include "inter_parameter.hpp"
#include "reduce.hpp"

#include <algorithm>

InterParameter::InterParameter(std::shared_ptr<const System> system,
                               real value, std::string name, std::string unit)
    : system_(system),
      name_(name),
      unit_(unit),
      valuesbuffer_() {
  std::vector<uint> uni = system->uniqueRegions;
  if (!uni.empty()) {
    size_t N = *std::max_element(uni.begin(), uni.end()) + 1;
    valuesLimit_ = N*(N-1)/2;
    valuesbuffer_ = GpuBuffer<real>(std::vector<real>(valuesLimit_, value));
  }
}

const GpuBuffer<real>& InterParameter::values() const {
  return valuesbuffer_;
}

__global__ void k_set(real* values, real value) {
  // TODO: more efficient, memory-wise, to only set relevant elements?
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  values[idx] = value;
}

void InterParameter::set(real value) {
  cudaLaunch(valuesLimit_, k_set, valuesbuffer_.get(), value);
}

__global__ void k_setBetween(real* values, int index, real value) {
  values[index] = value;
}

void InterParameter::setBetween(uint i, uint j, real value) {
  system_->checkIdxInRegions(i);
  system_->checkIdxInRegions(j);
  if (i == j) {
    throw std::invalid_argument("Can not set " + name_
            + ", region indexes must be different: " + std::to_string(i) + ".");
  }

  // Update GPU memory directly
  cudaLaunchReductionKernel(k_setBetween, valuesbuffer_.get(), getLutIndex(i, j), value);
  return;
}

CuInterParameter InterParameter::cu() const {
  return CuInterParameter(this);
}