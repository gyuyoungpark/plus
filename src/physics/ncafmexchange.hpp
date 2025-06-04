#pragma once

#include "quantityevaluator.hpp"

// returns the deviation from the optimal angle (120°) between magnetization
// vectors in the same cell which are coupled by the intracell exchange interaction.
Field evalAngleField(const NCAFM*);
// The maximal angle between sublattice spins.
real evalMaxAngle(const Ferromagnet*, const Ferromagnet*);

NCAFM_FieldQuantity angleFieldQuantity(const NCAFM*);