#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;


Field evalStressTensor(const Ferromagnet*);

// Stress tensor quantity with 6 symmetric stress components  
// [σxx, σyy, σzz, σxy, σxz, σyz],  
// calculated according to σ = c ε. 
FM_FieldQuantity stressTensorQuantity(const Ferromagnet*);
