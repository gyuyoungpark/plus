#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;


bool elasticDampingAssuredZero(const Ferromagnet*);
bool kineticEnergyAssuredZero(const Ferromagnet*)

Field evalElasticDamping(const Ferromagnet*);
Field evalEffectiveBodyForce(const Ferromagnet*);
Field evalElasticAcceleration(const Ferromagnet*);
Field evalKineticEnergyDensity(const Ferromagnet*);
Field evalElasticEnergyDensity(const Ferromagnet*);
Field evalStressTensor(const Ferromagnet*);

real kineticEnergy(const Ferromagnet*);
real elasticEnergy(const Ferromagnet*);

// Elastic damping proportional to η and velocity: -ηv.
FM_FieldQuantity elasticDampingQuantity(const Ferromagnet*);

// Elastic effective body force is the sum of elastic, magnetoelastic and
// external body forces. Elastic damping is not included.
FM_FieldQuantity effectiveBodyForceQuantity(const Ferromagnet*);

// Translate const Variable* elasticVelocity to usable FM_fieldQuantity
FM_FieldQuantity elasticVelocityQuantity(const Ferromagnet*);

// Elastic acceleration includes all effects that influence the elastic velocity
// including elastic, magnetoelastic and external body forces, and elastic damping.
FM_FieldQuantity elasticAccelerationQuantity(const Ferromagnet*);

// stress tensor
FM_FieldQuantity stressTensorQuantity(const Ferromagnet*);


// kinetic and elastic energy
FM_FieldQuantity kineticEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity kineticEnergyQuantity(const Ferromagnet*);

FM_FieldQuantity elasticEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity elasticEnergyQuantity(const Ferromagnet*);