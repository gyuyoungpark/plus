#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;

bool viscousDampingAssuredZero(const Magnet*);

Field evalElasticStress(const Magnet*);
Field evalViscousStress(const Magnet*);
Field evalStressTensor(const Magnet*);

// Elastic stress tensor quantity with 6 symmetric stress components  
// calculated according to σ = c:ε. 
M_FieldQuantity elasticStressQuantity(const Magnet*);
// Viscous stress tensor quantity with 6 symmetric stress components  
// calculated according to σ = η_b vol(dε/dt) + η_ν dev(dε/dt).
M_FieldQuantity viscousStressQuantity(const Magnet*);
// Total stress tensor quantity with 6 symmetric stress components  
// [σxx, σyy, σzz, σxy, σxz, σyz]
M_FieldQuantity stressTensorQuantity(const Magnet*);
