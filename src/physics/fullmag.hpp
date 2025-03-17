#pragma once

#include "quantityevaluator.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;
class NCAFM;

Field evalFullMag(const Ferromagnet*);
Field evalAFMFullMag(const Antiferromagnet*);
Field evalNCAFMFullMag(const NCAFM*);

FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet*);
AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet*);
NCAFM_FieldQuantity fullMagnetizationQuantity(const NCAFM*);