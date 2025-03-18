#pragma once

#include "quantityevaluator.hpp"

class NCAFM;
class Field;

Field evalOctupoleVector(const NCAFM*);
NCAFM_FieldQuantity octupoleVectorQuantity(const NCAFM*);