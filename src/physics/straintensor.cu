#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"


bool strainTensorAssuredZero(const Ferromagnet* magnet) {
  return !magnet->getEnableElastodynamics();
}


// TODO: add k_strainTensor() {}


Field evalStrainTensor(const Ferromagnet* magnet) {
  Field strain(magnet->system(), 6, 0.0);

  if (strainTensorAssuredZero(magnet))
    return strain;

  // TODO:
  // auto property = magnet->property;
  // cudalaunch(k_strainTensor, properties)
  return strain;
}


FM_FieldQuantity strainTensorQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalStrainTensor, 6, "strain", "");
}
