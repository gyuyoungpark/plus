#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

// Spin-rotation coupling (Barnett effect):
// H_Barnett = omega / gamma  (in code units, Tesla)
// where omega = 1/2 curl(v) is the lattice angular velocity.
bool spinRotationAssuredZero(const Ferromagnet*);

Field evalSpinRotationField(const Ferromagnet*);
Field evalSpinRotationEnergyDensity(const Ferromagnet*);
real evalSpinRotationEnergy(const Ferromagnet*);

FM_FieldQuantity spinRotationFieldQuantity(const Ferromagnet*);
FM_FieldQuantity spinRotationEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity spinRotationEnergyQuantity(const Ferromagnet*);
