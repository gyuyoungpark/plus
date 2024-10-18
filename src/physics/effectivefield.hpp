#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

Field evalEffectiveField(const Ferromagnet*);
FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet*);
