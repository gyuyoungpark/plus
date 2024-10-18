#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;
class Magnet;

// Returns the energy density of an effective field term.
//   edens = - prefactor * Msat * dot(m,h)
// The prefactor depends on the origin of the effective field term
Field evalEnergyDensity(const Ferromagnet*, const Field&, real prefactor);

Field evalTotalEnergyDensity(const Ferromagnet*);
real evalTotalEnergy(const Magnet*);

FM_FieldQuantity totalEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity totalEnergyQuantity(const Ferromagnet*);
