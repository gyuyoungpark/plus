#include "elasticforce.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"


bool elasticForceAssuredZero(const Ferromagnet* magnet) {
  return ((!magnet->enableElastodynamics()) ||
          (magnet->c11.assuredZero() && magnet->c12.assuredZero() &&
           magnet->c44.assuredZero()));
}


/** Returns index of coo+relcoo if that is inside the geometry.
 * Otherwise it returns index of coo itself (assumed to be safe).
 * This mimics open boundary conditions.
*/
__device__ int coord2safeIndex(int3 coo, int3 relcoo,
                               const CuSystem& system, const Grid& mastergrid) {
  const Grid grid = system.grid;
  int3 coo_ = mastergrid.wrap(coo + relcoo);
  if (grid.cellInGrid(coo_)) {  // don't convert to index if outside grid!
    int idx_ = grid.coord2index(coo_);
    if (system.inGeometry(idx_))
      return idx_;
  }
  return grid.coord2index(coo);
}

/** Returns index of coo+relcoo1+relcoo2 if that is inside the geometry.
 * Otherwise returns index of coo+relcoo1 (first) or coo+relcoo2 (second)
 * if one of those is inside the geometry.
 * Or it returns index of coo itself (assumed to be safe).
 * This mimics open boundary conditions.
*/
__device__ int coord2safeIndex(int3 coo, int3 relcoo1, int3 relcoo2,
                               const CuSystem& system, const Grid& mastergrid) {
  const Grid grid = system.grid;
  int3 coo_[3] = {mastergrid.wrap(coo + relcoo1 + relcoo2),
                  mastergrid.wrap(coo + relcoo1), mastergrid.wrap(coo + relcoo2)};
  for (int i = 0; i < 3; i++) {
    if (grid.cellInGrid(coo_[i])) {  // don't convert to index if outside grid!
      int idx_ = grid.coord2index(coo_[i]);
      if (system.inGeometry(idx_))
        return idx_;
    }
  }
  return grid.coord2index(coo);
}

// ∂i(c ∂i(u))
// position index due to derivative, not component index!
// w = 1/cellsize
__device__ real doubleDerivative(real c_im1, real c_i, real c_ip1,
                                 real u_im1, real u_i, real u_ip1, real wi) {
  return (  harmonicMean(c_ip1, c_i) * (u_ip1 - u_i  )
          - harmonicMean(c_i, c_im1) * (u_i   - u_im1)) * (wi*wi);
}

// ∂j(c ∂i(u))
// ≈ ∂j(c)∂i(u) + c ∂j∂i(u)
// position index due to derivative, not component index!
// w = 1/cellsize
__device__ real mixedDerivative(real c_i_jm1, real c_i_j, real c_i_jp1,
                                real u_im1_j, real u_ip1_j,
                                real u_im1_jm1, real u_im1_jp1,
                                real u_ip1_jm1, real u_ip1_jp1,
                                real wi, real wj) {
  real f = (c_i_jp1 - c_i_jm1) * (u_ip1_j - u_im1_j);  // ~ ∂j(c)∂i(u)
  // optimal order of terms to minimize numerical noise for 2D materials (nz = 1)
  f += c_i_j * (u_ip1_jp1 - u_im1_jp1 - u_ip1_jm1 + u_im1_jm1);  // ~ c ∂j∂i(u)
  f *= 0.25 * wi * wj;
  return f;
}


// I tried to adhere to openBC==true using safeCoo
// similar to openBC in k_exchangeField (using continue)
// TODO: but need friction-free boundary conditions te be correct
__global__ void k_elasticForce(CuField fField,
                               const CuField uField,
                               const CuParameter c11,
                               const CuParameter c12,
                               const CuParameter c44,
                               const real3 w,
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

  // array instead of real3 to get indexing [i]
  const real ws[3] = {w.x, w.y, w.z};
  const int gs[3] = {grid.size().x, grid.size().y, grid.size().z};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0,-1, 0}, int3{0, 0,-1}};
  const int3 coo = grid.index2coord(idx);

#pragma unroll  // TODO: check if faster with or without this unrolling
  for (int i = 0; i < 3; i++) {  // i is a {x, y, z} component/direction
    real f_i = 0;  // force component i

    // =============================================
    // f_i += ∂i(c11 ∂i(u_i))    so c11 with ∂²_i u_i
    // take u component i
    // translate all in direction i

    real ui = uField.valueAt(idx, i);
    real wi = ws[i];
    int3 im1 = im1_arr[i], ip1 = ip1_arr[i];  // transl in direction i
    int safeIdx_im1 = coord2safeIndex(coo, im1, system, mastergrid);
    int safeIdx_ip1 = coord2safeIndex(coo, ip1, system, mastergrid);
    if (gs[i] > 1)  // only derivative calculation if more than 1 cell
      f_i += doubleDerivative(c11.valueAt(safeIdx_im1),
                              c11.valueAt(idx),
                              c11.valueAt(safeIdx_ip1),
                              uField.valueAt(safeIdx_im1, i), ui,
                              uField.valueAt(safeIdx_ip1, i), wi);

#pragma unroll  // TODO: check if faster with or without this unrolling
    for (int j_=i+1; j_<i+3; j_++) {
      // j is one of the *other* {x, y, z} components/directions
      int j = j_;
      if (j > 2) {j -= 3;};

      if (gs[j] <= 1) continue;  // no derivative calculation if only 1 cell

      // translate in direction j
      real wj = ws[j];
      int3 jm1 = im1_arr[j], jp1 = ip1_arr[j];
      int safeIdx_jm1 = coord2safeIndex(coo, jm1, system, mastergrid);
      int safeIdx_jp1 = coord2safeIndex(coo, jp1, system, mastergrid);
      
      real c44_jm1 = c44.valueAt(safeIdx_jm1);
      real c44_    = c44.valueAt(idx);
      real c44_jp1 = c44.valueAt(safeIdx_jp1);

      // =============================================
      // f_i += ∂j(c44 ∂j(u_i))    so c44 with ∂²_j u_i
      // take u component i
      // translate all in j direction
      f_i += doubleDerivative(c44_jm1, c44_, c44_jp1,
                              uField.valueAt(safeIdx_jm1, i), ui,
                              uField.valueAt(safeIdx_jp1, i), wj);

      // ===========================================================
      // f_i += ∂j((c12+c44) ∂i(u_j))    so (c12+c44) with ∂_j∂_i u_j
      // translate c12+c44 in j direction
      // take u component j
      // translate u in both i and j directions

      if (gs[i] <= 1) continue;  // no derivative calculation if only 1 cell

      // translate in both i and j directions
      int safeIdx_im1_jm1 = coord2safeIndex(coo, im1, jm1, system, mastergrid);
      int safeIdx_im1_jp1 = coord2safeIndex(coo, im1, jp1, system, mastergrid);
      int safeIdx_ip1_jm1 = coord2safeIndex(coo, ip1, jm1, system, mastergrid);
      int safeIdx_ip1_jp1 = coord2safeIndex(coo, ip1, jp1, system, mastergrid);

      f_i += mixedDerivative(c12.valueAt(safeIdx_jm1) + c44_jm1,
                             c12.valueAt(idx) + c44_,
                             c12.valueAt(safeIdx_jp1) + c44_jp1,
                             uField.valueAt(safeIdx_im1, j),
                             uField.valueAt(safeIdx_ip1, j),
                             uField.valueAt(safeIdx_im1_jm1, j),
                             uField.valueAt(safeIdx_im1_jp1, j),
                             uField.valueAt(safeIdx_ip1_jm1, j),
                             uField.valueAt(safeIdx_ip1_jp1, j), wi, wj);
    }
    fField.setValueInCell(idx, i, f_i);
  }
}


Field evalElasticForce(const Ferromagnet* magnet) {

  Field fField(magnet->system(), 3);
  if (elasticForceAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }

  int ncells = fField.grid().ncells();
  CuField uField = magnet->elasticDisplacement()->field().cu();
  CuParameter c11 = magnet->c11.cu();
  CuParameter c12 = magnet->c12.cu();
  CuParameter c44 = magnet->c44.cu();
  real3 w = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(ncells, k_elasticForce, fField.cu(), uField, c11, c12, c44, w,
             mastergrid);

  return fField;
}

FM_FieldQuantity elasticForceQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalElasticForce, 3, "elastic_force", "N/m3");
}
