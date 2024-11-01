#pragma once

#include "quantityevaluator.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

Field evalFullMag(const Ferromagnet*);
Field evalAFMFullMag(const Antiferromagnet*);

FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet*);
AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet*);
