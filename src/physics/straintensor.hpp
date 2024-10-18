#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;


bool strainTensorAssuredZero(const Ferromagnet*);

Field evalStrainTensor(const Ferromagnet*);

// Strain tensor quantity with 6 symmetric strain components
// [εxx, εyy, εzz, εxy, εxz, εyz],
// calculated according to ε = 1/2 (∇u + (∇u)^T).
FM_FieldQuantity strainTensorQuantity(const Ferromagnet*);
