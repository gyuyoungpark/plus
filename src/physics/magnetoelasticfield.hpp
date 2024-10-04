#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;


// Assure that the magnetoelastic field and force are 0
bool magnetoelasticAssuredZero(const Ferromagnet*);

Field evalMagnetoelasticField(const Ferromagnet*);

// Magnetoelastic effective field due to effects of inverse magnetostriction
FM_FieldQuantity magnetoelasticFieldQuantity(const Ferromagnet*);
