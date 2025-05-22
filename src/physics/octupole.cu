#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "field.hpp"
#include "ncafm.hpp"
#include "octupole.hpp"

__device__ real3 rotate_120(real3 m, real3 k, real direction) {
  // rotate m about k over 120 degrees
  const real SQRT3_2 = 0.86602540378;
  return  -0.5 * m + cross(k, m) * SQRT3_2 * direction;
}

__global__ void k_octupolevector(CuField octupole,
                             const CuField mag1,
                             const CuField mag2,
                             const CuField mag3,
                             const CuParameter msat1,
                             const CuParameter msat2,
                             const CuParameter msat3) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!octupole.cellInGeometry(idx)) {
    if (octupole.cellInGrid(idx)) 
      octupole.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m1 = mag1.vectorAt(idx);
  real3 m2 = mag2.vectorAt(idx);
  real3 m3 = mag3.vectorAt(idx);
  real ms1 = msat1.valueAt(idx);
  real ms2 = msat2.valueAt(idx);
  real ms3 = msat3.valueAt(idx);

  real3 k2 = normalized(cross(m1, m2)); 
  real3 k3 = normalized(cross(m1, m3));

  real3 m2_rot = rotate_120(m2, k2, 1);
  real3 m3_rot = rotate_120(m3, k3, 1);

  // if (dot(m1, m2) < 0 is zelfde voorwaarde, dus veel simpeler)
  if (fabs(dot(m1, rotate_120(m2, k2, -1))) >= fabs(dot(m1, m2_rot)))
    // m2 rotated in wrong direction
    m2_rot = rotate_120(m2, k2, -1);

  if (fabs(dot(m1, rotate_120(m3, k3, -1))) >= fabs(dot(m1, m3_rot)))
    // m3 rotated in wrong direction
    m3_rot = rotate_120(m3, k3, -1);

  octupole.setVectorInCell(idx, (m1 * ms1 + m2_rot * ms2 + m3_rot * ms3) / (ms1 + ms2 + ms3));
}

Field evalOctupoleVector(const NCAFM* magnet) {
  // Calculate a weighted octupole vector (dimensionless) to account for NC-ferrimagnets
  Field octupole(magnet->system(), 3);

  if (magnet->sub1()->msat.assuredZero() &&
      magnet->sub2()->msat.assuredZero() &&
      magnet->sub3()->msat.assuredZero()) {
        octupole.makeZero();
        return octupole;
  }
  cudaLaunch(octupole.grid().ncells(), k_octupolevector, octupole.cu(),
             magnet->sub1()->magnetization()->field().cu(),
             magnet->sub2()->magnetization()->field().cu(),
             magnet->sub3()->magnetization()->field().cu(),
             magnet->sub1()->msat.cu(),
             magnet->sub2()->msat.cu(),
             magnet->sub3()->msat.cu());
  return octupole;
}

NCAFM_FieldQuantity octupoleVectorQuantity(const NCAFM* magnet) {
    return NCAFM_FieldQuantity(magnet, evalOctupoleVector, 3, "octupole_vector", "");
}