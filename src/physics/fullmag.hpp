#pragma once

#include "quantityevaluator.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;
class NCAFM;

Field evalFMFullMag(const Ferromagnet*);
Field evalHMFullMag(const HostMagnet*);

FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet*);
AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet*);
NCAFM_FieldQuantity fullMagnetizationQuantity(const NCAFM*);