#include "cudalaunch.hpp"
#include "elasticdamping.hpp"
#include "elasticforce.hpp"
#include "elastodynamics.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "magnetoelasticfield.hpp"
#include "magnetoelasticforce.hpp"
#include "parameter.hpp"


bool elasticityAssuredZero(const Ferromagnet* magnet) {
  return ((!magnet->enableElastodynamics()) ||
          (magnet->c11.assuredZero() && magnet->c12.assuredZero() &&
           magnet->c44.assuredZero()));
}

// ========== Effective Body Force ==========

Field evalEffectiveBodyForce(const Ferromagnet* magnet) {
  Field fField(magnet->system(), 3, 0.0);

  if (!elasticityAssuredZero(magnet))
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
  return FM_FieldQuantity(magnet,
       [](const Ferromagnet* magnet){return magnet->elasticVelocity()->eval();},
                          3, "elastic_velocity", "m/s");
}
