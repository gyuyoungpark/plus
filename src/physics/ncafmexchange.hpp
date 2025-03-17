#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;
class NCAFM;

// The homogeneous and inhomogeneous contributions to NCAFM exchange is split up.
// The homogeneous contribution (considering ncafmex_cell) corresponds to NCAFM
// exchange at a single site.
// The inhomogeneous contribution (considering ncafmex_nn) corresponds to NCAFM
// exchange between NN cells.

bool inHomoNCAfmExchangeAssuredZero(const Ferromagnet*);
bool homoNCAfmExchangeAssuredZero(const Ferromagnet*);

// Evaluate field
Field evalInHomogeneousNCAfmExchangeField(const Ferromagnet*);
Field evalHomogeneousNCAfmExchangeField(const Ferromagnet*);
// Evaluate energy density
Field evalInHomoNCAfmExchangeEnergyDensity(const Ferromagnet*);
Field evalHomoNCAfmExchangeEnergyDensity(const Ferromagnet*);
// Evaluate energy
real evalInHomoNCAfmExchangeEnergy(const Ferromagnet*);
real evalHomoNCAfmExchangeEnergy(const Ferromagnet*);

FM_FieldQuantity inHomoNCAfmExchangeFieldQuantity(const Ferromagnet*);
FM_FieldQuantity homoNCAfmExchangeFieldQuantity(const Ferromagnet*);

FM_FieldQuantity inHomoNCAfmExchangeEnergyDensityQuantity(const Ferromagnet*);
FM_FieldQuantity homoNCAfmExchangeEnergyDensityQuantity(const Ferromagnet*);

FM_ScalarQuantity inHomoNCAfmExchangeEnergyQuantity(const Ferromagnet*);
FM_ScalarQuantity homoNCAfmExchangeEnergyQuantity(const Ferromagnet*);

////////////////////////////////////////////////////////////////////////////////////

// returns the deviation from the optimal angle (120°) between magnetization
// vectors in the same cell which are coupled by the intracell exchange interaction.
Field evalAngleField(const NCAFM*);
// The maximal deviation from 120*.
real evalMaxAngle(const NCAFM*);

NCAFM_FieldQuantity angleFieldQuantity(const NCAFM*);
NCAFM_ScalarQuantity maxAngle(const NCAFM*);

