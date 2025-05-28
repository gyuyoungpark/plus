#include "ncafm.hpp"
#include "cudalaunch.hpp"
#include "ncafmexchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"

__global__ void k_angle(CuField angleField,
                        const CuField mField1,
                        const CuField mField2,
                        const CuField mField3,
                        const CuParameter ncafmex,
                        const CuParameter msat1,
                        const CuParameter msat2,
                        const CuParameter msat3) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!angleField.cellInGeometry(idx)) {
    if (angleField.cellInGrid(idx))
      angleField.setVectorInCell(idx, real3{0, 0, 0});
      return;
  }

  if (ncafmex.valueAt(idx) == 0) {
    angleField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  bool b1 = (msat1.valueAt(idx) != 0);
  bool b2 = (msat2.valueAt(idx) != 0);
  bool b3 = (msat3.valueAt(idx) != 0);

  real dev12 = acos(dot(mField1.vectorAt(idx) * b1, mField2.vectorAt(idx) * b2))
                    - (120.0 * M_PI / 180.0);
  real dev13 = acos(dot(mField1.vectorAt(idx) * b1, mField3.vectorAt(idx) * b3))
                    - (120.0 * M_PI / 180.0);
  real dev23 = acos(dot(mField2.vectorAt(idx) * b2, mField3.vectorAt(idx) * b3))
                    - (120.0 * M_PI / 180.0);

  angleField.setVectorInCell(idx, real3{dev12, dev13, dev23});
}

Field evalAngleField(const NCAFM* magnet) {
  // Three components for the angles between 1-2, 1-3 and 2-3
  Field angleField(magnet->system(), 3);

  cudaLaunch(angleField.grid().ncells(), k_angle, angleField.cu(),
             magnet->sub1()->magnetization()->field().cu(),
             magnet->sub2()->magnetization()->field().cu(),
             magnet->sub3()->magnetization()->field().cu(),
             magnet->afmex_cell.cu(),
             magnet->sub1()->msat.cu(),
             magnet->sub2()->msat.cu(),
             magnet->sub3()->msat.cu());
  return angleField;
}

real evalMaxAngle(const NCAFM* magnet) {
return maxAbsValue(evalAngleField(magnet));
}

NCAFM_FieldQuantity angleFieldQuantity(const NCAFM* magnet) {
return NCAFM_FieldQuantity(magnet, evalAngleField, 3, "angle_field", "rad");
}

NCAFM_ScalarQuantity maxAngle(const NCAFM* magnet) {
return NCAFM_ScalarQuantity(magnet, evalMaxAngle, "max_angle", "rad");
}