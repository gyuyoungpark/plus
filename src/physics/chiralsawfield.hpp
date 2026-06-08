#pragma once

#include "ferromagnet.hpp"
#include "quantityevaluator.hpp"

/**
 * @brief Chiral Surface Acoustic Wave (SAW) effective field.
 *
 * Computes the combined effective field from a chiral Rayleigh SAW
 * propagating in-plane, including three coupling channels:
 *
 *   1. Magnetoelastic (MEL): H_mel = (2*B1/Ms) * eps_xx(r,t) * diag(m)
 *   2. Magneto-rotation (MR): H_mr from K_mr and Omega_y(r,t)
 *   3. Barnett: H_B = omega_y(r,t) / gamma
 *
 * All three fields are computed in a single GPU kernel to ensure
 * phase consistency between channels and avoid Python callbacks.
 *
 * SAW fields:
 *   eps_xx(r,t) = eps0 * sin(k*x - omega*t + phi)
 *   Omega_y(r,t) = (xi*eps0/2) * cos(k*x - omega*t + phi)
 *   omega_y(r,t) = (xi*eps0*omega/2) * sin(k*x - omega*t + phi)
 *
 * Reference: Park, Lee, Shuai (2026), PRL.
 */

bool chiralSAWFieldAssuredZero(const Ferromagnet* magnet);

Field evalChiralSAWField(const Ferromagnet* magnet);

FM_FieldQuantity chiralSAWFieldQuantity(const Ferromagnet* magnet);
