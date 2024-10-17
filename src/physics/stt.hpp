#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

bool spinTransferTorqueAssuredZero(const Ferromagnet*);

bool ZhangLiSTTAssuredZero(const Ferromagnet*);

bool SlonczewskiSTTAssuredZero(const Ferromagnet*);

Field evalSpinTransferTorque(const Ferromagnet*);

FM_FieldQuantity spinTransferTorqueQuantity(const Ferromagnet*);
