#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Magnet;
class Field;

bool magnetoRotationAssuredZero(const Ferromagnet*);

Field evalRotationVector(const Magnet*);
Field evalMagnetoRotationField(const Ferromagnet*);
Field evalMagnetoRotationEnergyDensity(const Ferromagnet*);
real evalMagnetoRotationEnergy(const Ferromagnet*);

// Angular velocity = 1/2 curl(v), same math as rotation vector but on velocity
Field evalAngularVelocity(const Magnet*);

M_FieldQuantity rotationVectorQuantity(const Magnet*);
M_FieldQuantity angularVelocityQuantity(const Magnet*);
FM_FieldQuantity magnetoRotationFieldQuantity(const Ferromagnet*);
FM_FieldQuantity magnetoRotationEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity magnetoRotationEnergyQuantity(const Ferromagnet*);
