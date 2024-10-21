#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;


Field evalPoyntingVector(const Ferromagnet*);

// poynting vector
FM_FieldQuantity poyntingVectorQuantity(const Ferromagnet*);
