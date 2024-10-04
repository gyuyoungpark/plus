// TODO: check if these includes are really all necessary
#include "magnetoelastics.hpp"
#include "cudalaunch.hpp"
#include "elasticforce.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"
#include "magnetoelasticfield.hpp"  // magnetoelasticAssuredZero


// TODO:
// see https://github.com/Fredericvdv/Magnetoelasticity_MuMax3/blob/magnetoelastic/cuda/magnetoelasticforce.cu
Field evalMagnetoelasticForce(const Ferromagnet* magnet) {
  Field fField(magnet->system(), 3);
  if (magnetoelasticAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }

  // TODO:
  // auto property = magnet->property;
  // cudalaunch(k_magnetoelasticForce, properties);
  return fField;
}


FM_FieldQuantity magnetoelasticForceQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticForce, 3,
                          "magnetoelastic_force", "N/m3");
}
