#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;
class Magnet;

bool strayFieldsAssuredZero(const Ferromagnet*);
bool worldBiasFieldAssuredZero(const Magnet*);
bool magnetBiasFieldAssuredZero(const Ferromagnet*);
bool externalFieldAssuredZero(const Ferromagnet*);

Field evalExternalField(const Ferromagnet*);
Field evalZeemanEnergyDensity(const Ferromagnet*);
real zeemanEnergy(const Ferromagnet*);

FM_FieldQuantity externalFieldQuantity(const Ferromagnet*);
FM_FieldQuantity zeemanEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity zeemanEnergyQuantity(const Ferromagnet*);
