// TODO: check if these includes are really all necessary
#include "cudalaunch.hpp"
#include "elasticforce.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"


bool magnetoelasticAssuredZero(const Ferromagnet* magnet) {
  return ((!magnet->getEnableElastodynamics()) ||
          (magnet->msat.assuredZero()) ||
          (magnet->B1.assuredZero() && magnet->B2.assuredZero()));
}

// TODO:
// see https://github.com/Fredericvdv/Magnetoelasticity_MuMax3/blob/magnetoelastic/cuda/magnetoelasticfield.cu
Field evalMagnetoelasticField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (magnetoelasticAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  // TODO:
  // auto property = magnet->property;
  // cudalaunch(k_magnetoelasticField, properties);
  return hField;
}


FM_FieldQuantity magnetoelasticFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticField, 3,
                          "magnetoelastic_field", "T");
}
