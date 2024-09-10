#pragma once

#include <curand.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "dmitensor.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "poissonsystem.hpp"
#include "variable.hpp"
#include "world.hpp"
#include "system.hpp"

class Antiferromagnet;

class Ferromagnet : public Magnet {
 public:
  Ferromagnet(std::shared_ptr<System> system_ptr,
              std::string name,
              Antiferromagnet* hostMagnet_ = nullptr);

  Ferromagnet(MumaxWorld* world,
              Grid grid,
              std::string name,
              GpuBuffer<bool> geometry,
              GpuBuffer<uint> regions);
  ~Ferromagnet() override;

  const Variable* magnetization() const;

  bool isSublattice() const;
  const Antiferromagnet* hostMagnet() const;  // TODO: right amount of const?

  void minimize(real tol = 1e-6, int nSamples = 10);
  void relax(real tol);

  void setInterExchange(uint idx1, uint idx2, real value);

  std::tuple<std::vector<uint>, std::unordered_map<uint, uint>> constructIndexMap(std::vector<uint>);
  // Host and device functions to get LUT index
  int getLutIndex(int, int);
  //__device__ real getInterExchange(this*, uint i, uint j) const;

 private:
  NormalizedVariable magnetization_;

  // TODO: what type of pointer?
  // TODO: Magnet or Antiferromagnet?
  Antiferromagnet* hostMagnet_;

 public:
  mutable PoissonSystem poissonSystem;
  bool enableDemag;
  bool enableOpenBC;
  bool enableZhangLiTorque;
  bool enableSlonczewskiTorque;
  VectorParameter anisU;
  VectorParameter anisC1;
  VectorParameter anisC2;
  VectorParameter jcur;
  VectorParameter FixedLayer;
  /** Uniform bias magnetic field which will affect a ferromagnet.
   * Measured in Teslas.
   */
  VectorParameter biasMagneticField;
  Parameter msat;
  Parameter aex;
  Parameter ku1;
  Parameter ku2;
  Parameter kc1;
  Parameter kc2;
  Parameter kc3;
  Parameter alpha;
  Parameter temperature;
  Parameter Lambda;
  Parameter FreeLayerThickness;
  Parameter eps_prime;
  Parameter xi;
  Parameter pol;
  Parameter appliedPotential;
  Parameter conductivity;
  Parameter amrRatio;
  real RelaxTorqueThreshold;
  
  curandGenerator_t randomGenerator;

  DmiTensor dmiTensor;

  std::vector<uint> regIndices_;
  GpuBuffer<uint> regionIndices_;
  std::unordered_map<uint, uint> indexMap_;
  GpuBuffer<real> interExchange_;
  real* interExch_; // Device ptr of the gpu buffer
  uint* regPtr_;
};

__device__ inline real getInterExchange(uint idx1, uint idx2,
                                        real const* interEx, uint const* regPtr) {

  int i = findIndex(regPtr, idx1);
  int j = findIndex(regPtr, idx2);

  int index;
  if (i <= j)
    index = j * (j + 1) / 2 + i;
  else
    index = i * (i + 1) / 2 + j;

  return interEx[index];
}