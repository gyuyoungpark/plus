#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;


bool kineticEnergyAssuredZero(const Ferromagnet*);

Field evalKineticEnergyDensity(const Ferromagnet*);
Field evalElasticEnergyDensity(const Ferromagnet*);

real evalKineticEnergy(const Ferromagnet*);
real evalElasticEnergy(const Ferromagnet*);

// Elastic kinetic energy density
FM_FieldQuantity kineticEnergyDensityQuantity(const Ferromagnet*);
// Elastic kinetic energy
FM_ScalarQuantity kineticEnergyQuantity(const Ferromagnet*);

// Elastic potential energy density
FM_FieldQuantity elasticEnergyDensityQuantity(const Ferromagnet*);
// Elastic potential energy
FM_ScalarQuantity elasticEnergyQuantity(const Ferromagnet*);
