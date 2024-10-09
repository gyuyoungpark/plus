// TODO: check if these includes are really all necessary
#include "cudalaunch.hpp"
#include "elasticforce.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "magnetoelasticfield.hpp"  // magnetoelasticAssuredZero
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"


__global__ void k_magnetoelasticForce(CuField fField,
                                      const CuField m,
                                      const CuParameter B1,
                                      const CuParameter B2,
                                      const real3 cellsize,
                                      const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;
  const Grid grid = system.grid;
  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      fField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }
  const real cs[3] = {cellsize.x, cellsize.y, cellsize.z};
  const int3 im2_arr[3] = {int3{-2, 0, 0}, int3{0,-2, 0}, int3{0, 0,-2}};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0,-1, 0}, int3{0, 0,-1}};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  const int3 ip2_arr[3] = {int3{ 2, 0, 0}, int3{0, 2, 0}, int3{0, 0, 2}};
  const int3 coo = grid.index2coord(idx);

  real der[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};  // derivatives ∂j(mi)
  for (int i=0; i<3; i++) {  // i is a {x, y, z} component
    // take m component i
    real mi = m.valueAt(idx, i);
    for (int j=0; j<3; j++) {  // j is a {x, y, z} direction
      // take translation in j direction
      real dj = cs[j]; 
      int3 jm2 = im2_arr[j], jm1 = im1_arr[j];  // transl in direction -j
      int3 jp1 = ip1_arr[j], jp2 = ip2_arr[j];  // transl in direction +j
      
      int3 coo_jm2 = mastergrid.wrap(coo + jm2);
      int3 coo_jm1 = mastergrid.wrap(coo + jm1);
      int3 coo_jp1 = mastergrid.wrap(coo + jp1);
      int3 coo_jp2 = mastergrid.wrap(coo + jp2);
      
      // determine a derivative ∂j(m_i)
      real dmidj;
      if (!system.inGeometry(coo_jm1) && !system.inGeometry(coo_jp1)) {
        // --1-- zero
        dmidj = 0.f;
      } else if ((!system.inGeometry(coo_jm2) || !system.inGeometry(coo_jp2)) &&
                  system.inGeometry(coo_jm1) && system.inGeometry(coo_jp1)) {
        // -111-, 1111-, -1111 central difference,  ε ~ h^2
        dmidj = 0.5f * (m.valueAt(coo_jp1, i) - m.valueAt(coo_jm1, i));
      } else if (!system.inGeometry(coo_jm2) && !system.inGeometry(coo_jp1)) {
        // -11-- backward difference, ε ~ h^1
        dmidj =  (mi - m.valueAt(coo_jm1, i));
      } else if (!system.inGeometry(coo_jm1) && !system.inGeometry(coo_jp2)) {
        // --11- forward difference,  ε ~ h^1
        dmidj = (-mi + m.valueAt(coo_jp1, i));
      } else if (system.inGeometry(coo_jm2) && !system.inGeometry(coo_jp1)) {
        // 111-- backward difference, ε ~ h^2
        dmidj =  (0.5f * m.valueAt(coo_jm2, i) - 2.0f * m.valueAt(coo_jm1, i) + 1.5f * mi);
      } else if (!system.inGeometry(coo_jm1) && system.inGeometry(coo_jp1)) {
        // --111 forward difference,  ε ~ h^2
        dmidj = (-0.5f * m.valueAt(coo_jp2, i) + 2.0f * m.valueAt(coo_jp1, i) - 1.5f * mi);
      } else {
        // 11111 central difference,  ε ~ h^4
        dmidj = ((2.0f/3.0f)*(m.valueAt(coo_jp1, i) - m.valueAt(coo_jm1, i)) + 
                (1.0f/12.0f)*(m.valueAt(coo_jm2, i) - m.valueAt(coo_jp2, i)));
      }
      der[j][i] = dmidj / dj;
    }
  }
  for (int i=0; i<3; i++) {
    real m_here[3] = {m.valueAt(idx, 0), m.valueAt(idx, 1), m.valueAt(idx, 2)};
    int ip1 = i+1;
    int ip2 = i+2;

    // If they exceed 3, loop around
    if (ip1 >= 3){
      ip1 -= 3;
    } 
    if (ip2 >= 3){
      ip2 -= 3;
    }

    real f = 2 * B1.valueAt(idx) * m_here[i] * der[i][i];
    f += B2.valueAt(idx) * m_here[i] * (der[ip1][ip1] + der[ip2][ip2]);
    f += B2.valueAt(idx) * (m_here[ip1] * der[ip1][i] + m_here[ip2] * der[ip2][i]);
    fField.setValueInCell(idx, i, f);
  }
}


Field evalMagnetoelasticForce(const Ferromagnet* magnet) {
  Field fField(magnet->system(), 3);
  if (magnetoelasticAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }
  int ncells = fField.grid().ncells();
  CuField m = magnet->magnetization()->field().cu();
  CuParameter B1 = magnet->B1.cu();
  CuParameter B2 = magnet->B2.cu();
  real3 cellsize = magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();
  cudaLaunch(ncells, k_magnetoelasticForce, fField.cu(), m, B1, B2, cellsize, mastergrid);
  return fField;
}


FM_FieldQuantity magnetoelasticForceQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticForce, 3,
                          "magnetoelastic_force", "N/m3");
}
