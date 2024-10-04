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
#include "inter_parameter.hpp"
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
  // TODO: make displacement/velocity safe even if enableElastodynamics==false
  const Variable* elasticDisplacement() const;
  const Variable* elasticVelocity() const;

  bool isSublattice() const;
  const Antiferromagnet* hostMagnet() const;  // TODO: right amount of const?

  void minimize(real tol = 1e-6, int nSamples = 10);
  void relax(real tol);

 private:
  NormalizedVariable magnetization_;

  // these take a lot of memory. Don't initialize unless wanted!
  std::unique_ptr<Variable> elasticDisplacement_;
  std::unique_ptr<Variable> elasticVelocity_;
  bool enableElastodynamics_;  // TODO: or other name? enableMagnetoelastodynamics?

  // TODO: what type of pointer?
  // TODO: Magnet or Antiferromagnet?
  Antiferromagnet* hostMagnet_;

 public:
  mutable PoissonSystem poissonSystem;
  bool enableDemag;
  bool enableOpenBC;
  bool enableZhangLiTorque;
  bool enableSlonczewskiTorque;
  bool fixedLayerOnTop;
  bool getEnableElastodynamics() const {return enableElastodynamics_;}
  void setEnableElastodynamics(bool);
  VectorParameter anisU;
  VectorParameter anisC1;
  VectorParameter anisC2;
  VectorParameter jcur;
  VectorParameter fixedLayer;
  /** Uniform bias magnetic field which will affect a ferromagnet.
   * Measured in Teslas.
   */
  VectorParameter biasMagneticField;
  Parameter msat;
  Parameter aex;
  InterParameter interExch;
  InterParameter scaleExch;
  Parameter ku1;
  Parameter ku2;
  Parameter kc1;
  Parameter kc2;
  Parameter kc3;
  Parameter alpha;
  Parameter temperature;
  Parameter Lambda;
  Parameter freeLayerThickness;
  Parameter epsilonPrime;
  Parameter xi;
  Parameter pol;
  Parameter appliedPotential;
  Parameter conductivity;
  Parameter amrRatio;
  real RelaxTorqueThreshold;
  
  curandGenerator_t randomGenerator;

  DmiTensor dmiTensor;

  // Magnetoelasticity
  // TODO: should these be Magnet Parameters or Ferromagnet parameters?
  
  // stiffness constant; TODO: can this be generalized to 6x6 tensor?
  Parameter c11;  // c11 = c22 = c33
  Parameter c12;  // c12 = c13 = c23
  Parameter c44;  // c44 = c55 = c66

  Parameter eta;  // Phenomenological elastic damping constant
  Parameter rho;  // Mass density
  Parameter B1;  // First magnetoelastic coupling constant
  Parameter B2;  // Second magnetoelastic coupling constant

  // Members related to regions
  std::unordered_map<uint, uint> indexMap_;
  GpuBuffer<real> interExchange_;
  real* interExchPtr_ = nullptr; // Device pointer to interexch GpuBuffer
  uint* regPtr_ = nullptr; // Device pointer to GpuBuffer with unique region idxs
};
