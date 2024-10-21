#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;


bool elasticDampingAssuredZero(const Ferromagnet*);

Field evalElasticDamping(const Ferromagnet*);

// Elastic damping proportional to η and velocity: -ηv.
FM_FieldQuantity elasticDampingQuantity(const Ferromagnet*);
