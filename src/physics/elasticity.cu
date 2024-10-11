// TODO: check if these includes are really all necessary
#include "cudalaunch.hpp"
#include "elasticforce.hpp"
#include "elasticity.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "magnetoelasticfield.hpp"
#include "magnetoelasticforce.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"
#include "world.hpp"


// ========== Elastic Damping ==========

bool elasticDampingAssuredZero(const Ferromagnet* magnet) {
    return ((!magnet->getEnableElastodynamics()) || magnet->eta.assuredZero());
}

// Dedicated kernel function for -1 * eta * v; otherwise need two kernel calls.
__global__ void k_elasticDamping(CuField fField,
                                 const CuField vField,
                                 const CuParameter eta) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx)) {
      fField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  fField.setVectorInCell(idx, -eta.valueAt(idx) * vField.vectorAt(idx));
}

Field evalElasticDamping(const Ferromagnet* magnet) {
    Field fField(magnet->system(), 3);
    if (elasticDampingAssuredZero(magnet)) {
        fField.makeZero();
        return fField;
    }

    int ncells = fField.grid().ncells();
    CuField vField = magnet->elasticVelocity()->field().cu();
    CuParameter eta = magnet->eta.cu();

    cudaLaunch(ncells, k_elasticDamping, fField.cu(), vField, eta);

    return fField;
}

FM_FieldQuantity elasticDampingQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalElasticDamping, 3,
                          "elastic_damping", "N/m3");
}

// ========== Effective Body Force ==========

Field evalEffectiveBodyForce(const Ferromagnet* magnet) {
  Field fField(magnet->system(), 3, 0.0);

  if (!elasticForceAssuredZero(magnet))
    fField += evalElasticForce(magnet);
  if (!magnetoelasticAssuredZero(magnet))
    fField += evalMagnetoelasticForce(magnet);
  if (!magnet->externalBodyForce.assuredZero())
    fField += magnet->externalBodyForce;

  return fField;
}

FM_FieldQuantity effectiveBodyForceQuantity(const Ferromagnet* magnet) {
    return FM_FieldQuantity(magnet, evalEffectiveBodyForce, 3,
                            "effective_body_force", "N/m3");
}

// ========== Elastic Accelleration ==========

__global__ void k_divideByParam(CuField field, const CuParameter param) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = field.system;
  const Grid grid = system.grid;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      field.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  real p = param.valueAt(idx);
  if (p != 0) {
    field.setVectorInCell(idx, field.vectorAt(idx) / p);
  } else {
    field.setVectorInCell(idx, real3{0, 0, 0});  // substitue for infinity
  }
}

Field evalElasticAcceleration(const Ferromagnet* magnet) {
  Field aField = evalEffectiveBodyForce(magnet) + evalElasticDamping(magnet);
  
  // divide by rho if possible
  if (!magnet->rho.assuredZero())
    cudaLaunch(aField.grid().ncells(), k_divideByParam,
               aField.cu(), magnet->rho.cu());

  return aField;
}

FM_FieldQuantity elasticAccelerationQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalElasticAcceleration, 3,
                          "elastic_acceleration", "m/s2");
}

// ========== Elastic Velocity Quantity ==========

FM_FieldQuantity elasticVelocityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet,  // use lambda function
       [](const Ferromagnet* magnet){return magnet->elasticVelocity()->eval();},
                          3, "elastic_velocity", "m/s");
}

// ========== Stress Tensor ==========

__global__ void k_stress(CuField stressField,
                         const CuField strain,
                         const CuParameter c11,
                         const CuParameter c12,
                         const CuParameter c44) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = stressField.system;
  const Grid grid = system.grid;
  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      stressField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  for (int i=0; i<3; i++) {
    int ip1 = i+1;
    int ip2 = i+2;

    if (ip1 >= 3) {
      ip1 -= 3;
    }
    if (ip2 >= 3) {
      ip2 -= 3;
    }

    stressField.setValueInCell(idx, i,
                               c11.valueAt(idx) * strain.valueAt(idx, i) +
                               c12.valueAt(idx) * strain.valueAt(idx, ip1) +
                               c12.valueAt(idx) * strain.valueAt(idx, ip2));
    
    stressField.setValueInCell(idx, i+3, c44.valueAt(idx) * strain.valueAt(idx, i+3));
  }
}

Field evalStressTensor(const Ferromagnet* magnet) {
  Field stressField(magnet->system(), 6, 0.0);
  if (strainTensorAssuredZero(magnet)) return stressField;

  int ncells = stressField.grid().ncells();
  CuField strain = evalStrainTensor(magnet).cu();
  CuParameter c11 = magnet->c11.cu();
  CuParameter c12 = magnet->c12.cu();
  CuParameter c44 = magnet->c44.cu();

  cudaLaunch(ncells, k_stress, stressField.cu(), strain, c11, c12, c44);
  return stressField;
}

FM_FieldQuantity stressTensorQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalStressTensor, 6, "stress", "N/m2");
}

// ========== Kinetic Energy ==========

__global__ void k_kineticEnergy(CuField kinField,
                                const CuField velocity,
                                const CuParameter rho) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = kinField.system;
  const Grid grid = system.grid;
  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      kinField.setValueInCell(idx, 0, 0);
    }
    return;
  }

  real3 v = velocity.vectorAt(idx);
  real vel = v.x * v.x + v.y * v.y + v.z * v.z;

  kinField.setValueInCell(idx, 0, 0.5 * vel * rho.valueAt(idx));
}

Field evalKineticEnergyDensity(const Ferromagnet* magnet) {
  Field kinField(magnet->system(), 1);
  if (!magnet->getEnableElastodynamics()) {
    kinField.makeZero();
    return kinField;
  }

  int ncells = kinField.grid().ncells();
  CuField velocity = magnet->elasticVelocity()->field().cu();
  CuParameter rho = magnet->rho.cu();
  cudaLaunch(ncells, k_kineticEnergy, kinField.cu(), velocity, rho);
  return kinField;
}

real kineticEnergy(const Ferromagnet* magnet) {
  if (!magnet->getEnableElastodynamics())
    return 0.0;

  real edens = kineticEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity kineticEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalKineticEnergyDensity, 1, "kinetic_energy_density", "J/m3");
}

FM_ScalarQuantity kineticEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, kineticEnergy, "kinetic_energy", "J");
}

// ========== Elastic Energy ==========

__global__ void k_elasticEnergy(CuField elField,
                                  const CuField stress,
                                  const CuField strain) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = elField.system;
  const Grid grid = system.grid;
  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      elField.setValueInCell(idx, 0, 0);
    }
    return;
  }

  real value = 0;
  for (int i=0; i<3; i++){
    value += 0.5 * stress.valueAt(idx, i) * strain.valueAt(idx, i);
    value += stress.valueAt(idx, i+3) * strain.valueAt(idx, i+3);
  }

  elField.setValueInCell(idx, 0, value);
}

Field evalElasticEnergyDensity(const Ferromagnet* magnet) {
  Field elField(magnet->system(), 1);
  if (!magnet->getEnableElastodynamics()) {
    elField.makeZero();
  return elField;
  }

  int ncells = elField.grid().ncells();
  CuField stress = evalStressTensor(magnet).cu();
  CuField strain = evalStrainTensor(magnet).cu();
  cudaLaunch(ncells, k_elasticEnergy, elField.cu(), stress, strain);
  return elField;
}

real elasticEnergy(const Ferromagnet* magnet) {
  if (!magnet->getEnableElastodynamics())
    return 0.0;

  real edens = elasticEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity elasticEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalElasticEnergyDensity, 1, "elastic_energy_density", "J/m3");
}

FM_ScalarQuantity elasticEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, elasticEnergy, "elastic_energy", "J");
}