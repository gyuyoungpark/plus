#include "ferromagnet.hpp"

#include <curand.h>
#include <chrono>

#include <memory>
#include <random>
#include <math.h>
#include <cfloat>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "poissonsystem.hpp"
#include "relaxer.hpp"

Ferromagnet::Ferromagnet(std::shared_ptr<System> system_ptr, 
                         std::string name,
                         Antiferromagnet* hostMagnet)
    : Magnet(system_ptr, name),
      hostMagnet_(hostMagnet),
      magnetization_(name + ":magnetization", "", system(), 3),
      msat(system(), 1.0),
      aex(system(), 0.0),
      ku1(system(), 0.0),
      ku2(system(), 0.0),
      kc1(system(), 0.0),
      kc2(system(), 0.0),
      kc3(system(), 0.0),
      alpha(system(), 0.0),
      temperature(system(), 0.0),
      xi(system(), 0.0),
      Lambda(system(), 0.0),
      FreeLayerThickness(system(), grid().size().z * cellsize().z),
      eps_prime(system(), 0.0),
      FixedLayer(system(), {0, 0, 0}),
      pol(system(), 0.0),
      anisU(system(), {0, 0, 0}),
      anisC1(system(), {0, 0, 0}),
      anisC2(system(), {0, 0, 0}),
      jcur(system(), {0, 0, 0}),
      biasMagneticField(system(), {0, 0, 0}),
      dmiTensor(system()),
      enableDemag(true),
      enableOpenBC(false),
      enableZhangLiTorque(true),
      enableSlonczewskiTorque(true),
      appliedPotential(system(), std::nanf("0")),
      conductivity(system(), 0.0),
      amrRatio(system(), 0.0),
      RelaxTorqueThreshold(-1.0),
      poissonSystem(this) {
    {// Initialize random magnetization
    // TODO: this can be done much more efficient somewhere else
    int nvalues = 3 * this->grid().ncells();
    std::vector<real> randomValues(nvalues);
    std::normal_distribution<real> dist(0.0, 1.0);
    std::default_random_engine randomEngine;
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    randomEngine.seed (seed);
    for (auto& v : randomValues) {
      v = dist(randomEngine);
    }
    Field randomField(system(), 3);
    randomField.setData(&randomValues[0]);
    magnetization_.set(randomField);
  }
  {// Initialize interregional exchange buffer
    if (system()->regions().get()) {

      std::vector<uint> rIdxs;
      std::tie(rIdxs, indexMap_) = constructIndexMap(system()->regions().getData());
      GpuBuffer<uint> regionIndices_ = GpuBuffer<uint>(rIdxs);

      int N = rIdxs.size();
      std::vector<real> data(N * (N + 1) / 2, 0);

      interExchange_ = GpuBuffer<real>(data);
      interExch_ = interExchange_.get();
      regPtr_ = regionIndices_.get();
    }
  }
  // Initialize CUDA RNG
  // TODO: move the generator to somewhere else
  curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(randomGenerator,
        static_cast<int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  }

Ferromagnet::Ferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry,
                         GpuBuffer<uint> regions)
    : Ferromagnet(std::make_shared<System>(world, grid, geometry, regions), name) {}

Ferromagnet::~Ferromagnet() {
  curandDestroyGenerator(randomGenerator);
}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

bool Ferromagnet::isSublattice() const {
  return !(hostMagnet_ == nullptr);
}

const Antiferromagnet* Ferromagnet::hostMagnet() const {
  return hostMagnet_;
}

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void Ferromagnet::relax(real tol) {
  real threshold = this->RelaxTorqueThreshold;
  Relaxer relaxer(this, {threshold}, tol);
  relaxer.exec();
}

void Ferromagnet::setInterExchange(uint idx1, uint idx2, real value) {

  system()->checkIdxInRegions(idx1);
  system()->checkIdxInRegions(idx2);

  std::vector<real> interEx = interExchange_.getData(); // Copy data from GPU to host
  interEx[getLutIndex(indexMap_[idx1], indexMap_[idx2])] = value;
  
  interExchange_ = std::move(GpuBuffer<real>(interEx)); // Move back to old GpuBuffer
  interExch_ = interExchange_.get(); // Update pointer to buffer

  return;
}

std::tuple<std::vector<uint>, std::unordered_map<uint, uint>>
            Ferromagnet::constructIndexMap(std::vector<uint> regionsInGrid) {
  // Returns unique region indices and their position inside this container.
  std::vector<uint> regions;            
  std::unordered_map<uint, uint> indexMap;

  for (const auto& reg : regionsInGrid) {
    if (indexMap.find(reg) == indexMap.end()) {
      regions.push_back(reg);
      indexMap[reg] = regions.size() - 1;
    }
  }
  return std::make_tuple(regions, indexMap);
}