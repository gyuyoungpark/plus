#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

bool elasticForceAssuredZero(const Ferromagnet*);

Field evalElasticForce(const Ferromagnet*);

// Elastic body force due to mechanical stress gradients f = ∇σ = ∇(cε)
FM_FieldQuantity elasticForceQuantity(const Ferromagnet*);
