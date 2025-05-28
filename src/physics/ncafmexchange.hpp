#pragma once

#include "quantityevaluator.hpp"

// returns the deviation from the optimal angle (120°) between magnetization
// vectors in the same cell which are coupled by the intracell exchange interaction.
Field evalAngleField(const NCAFM*);
// The maximal deviation from 120*.
real evalMaxAngle(const NCAFM*);

NCAFM_FieldQuantity angleFieldQuantity(const NCAFM*);
NCAFM_ScalarQuantity maxAngle(const NCAFM*);

