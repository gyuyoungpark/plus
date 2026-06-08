#include "constants.hpp"
#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "magnet.hpp"
#include "magnetorotationfield.hpp"
#include "parameter.hpp"


bool magnetoRotationAssuredZero(const Ferromagnet* magnet) {
  // Check if elastodynamics OR prescribed rigid rotation is available
  bool hasRotationSource;
  if (magnet->isSublattice()) {
    hasRotationSource = magnet->hostMagnet()->enableElastodynamics() ||
                        !magnet->hostMagnet()->rigidRotation.assuredZero();
  } else {
    hasRotationSource = magnet->enableElastodynamics() ||
                        !magnet->rigidRotation.assuredZero();
  }

  return (!hasRotationSource || magnet->msat.assuredZero() ||
          magnet->Kmr.assuredZero());
}


// Compute the rotation pseudovector Omega = 1/2 curl(u)
__global__ void k_rotationVector(CuField rot,
                                 const CuField u,
                                 const real3 w,  // w = 1/cellsize
                                 const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = rot.system;
  const Grid grid = system.grid;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      rot.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  const real ws[3] = {w.x, w.y, w.z};
  const int3 im2_arr[3] = {int3{-2, 0, 0}, int3{0,-2, 0}, int3{0, 0,-2}};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0,-1, 0}, int3{0, 0,-1}};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  const int3 ip2_arr[3] = {int3{ 2, 0, 0}, int3{0, 2, 0}, int3{0, 0, 2}};
  const int3 coo = grid.index2coord(idx);

  real der[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};  // der[i][j] = d(u_j)/d(x_i)
  real3 u_0 = u.vectorAt(idx);
#pragma unroll
  for (int i = 0; i < 3; i++) {
    real wi = ws[i];
    int3 im2 = im2_arr[i], im1 = im1_arr[i];
    int3 ip1 = ip1_arr[i], ip2 = ip2_arr[i];

    int3 coo_im2 = mastergrid.wrap(coo + im2);
    int3 coo_im1 = mastergrid.wrap(coo + im1);
    int3 coo_ip1 = mastergrid.wrap(coo + ip1);
    int3 coo_ip2 = mastergrid.wrap(coo + ip2);

    real3 dudi;
    if (!system.inGeometry(coo_im1) && !system.inGeometry(coo_ip1)) {
      dudi = real3{0, 0, 0};
    } else if ((!system.inGeometry(coo_im2) ||
                !system.inGeometry(coo_ip2)) &&
                system.inGeometry(coo_im1) &&
                system.inGeometry(coo_ip1)) {
      dudi = 0.5 * (u.vectorAt(coo_ip1) - u.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im2) &&
               !system.inGeometry(coo_ip1)) {
      dudi = (u_0 - u.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im1) &&
               !system.inGeometry(coo_ip2)) {
      dudi = (-u_0 + u.vectorAt(coo_ip1));
    } else if (system.inGeometry(coo_im2) &&
               !system.inGeometry(coo_ip1)) {
      dudi = (0.5 * u.vectorAt(coo_im2) - 2.0 * u.vectorAt(coo_im1) +
              1.5 * u_0);
    } else if (!system.inGeometry(coo_im1) &&
                system.inGeometry(coo_ip1)) {
      dudi = (-0.5 * u.vectorAt(coo_ip2) + 2.0 * u.vectorAt(coo_ip1) -
              1.5 * u_0);
    } else {
      dudi = ((2.0/3.0) * (u.vectorAt(coo_ip1) - u.vectorAt(coo_im1)) +
              (1.0/12.0) * (u.vectorAt(coo_im2) - u.vectorAt(coo_ip2)));
    }
    dudi *= wi;

    der[i][0] = dudi.x;
    der[i][1] = dudi.y;
    der[i][2] = dudi.z;
  }

  // Extract antisymmetric part: Omega = 1/2 curl(u)
  // Omega_x = 1/2 (du_z/dy - du_y/dz) = 1/2 (der[1][2] - der[2][1])
  // Omega_y = 1/2 (du_x/dz - du_z/dx) = 1/2 (der[2][0] - der[0][2])
  // Omega_z = 1/2 (du_y/dx - du_x/dy) = 1/2 (der[0][1] - der[1][0])
  real Ox = 0.5 * (der[1][2] - der[2][1]);
  real Oy = 0.5 * (der[2][0] - der[0][2]);
  real Oz = 0.5 * (der[0][1] - der[1][0]);

  rot.setVectorInCell(idx, real3{Ox, Oy, Oz});
}


// Copy a prescribed VectorParameter into a Field
__global__ void k_prescribedVectorField(CuField out,
                                        const CuVectorParameter prescribed) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = out.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx))
      out.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
  out.setVectorInCell(idx, prescribed.vectorAt(idx));
}


Field evalRotationVector(const Magnet* magnet) {
  Field rot(magnet->system(), 3);

  // 1. If prescribed rigid rotation is set, use it (no elastodynamics needed)
  if (!magnet->rigidRotation.assuredZero()) {
    int ncells = rot.grid().ncells();
    CuVectorParameter prescribed = magnet->rigidRotation.cu();
    cudaLaunch(ncells, k_prescribedVectorField, rot.cu(), prescribed);
    return rot;
  }

  // 2. Otherwise compute from elastodynamics
  if (!magnet->enableElastodynamics()) {
    rot.makeZero();
    return rot;
  }

  int ncells = rot.grid().ncells();
  CuField u = magnet->elasticDisplacement()->field().cu();
  real3 w = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(ncells, k_rotationVector, rot.cu(), u, w, mastergrid);
  return rot;
}


// Angular velocity omega = 1/2 curl(v), same math as rotation vector
// but applied to velocity field instead of displacement.
Field evalAngularVelocity(const Magnet* magnet) {
  Field omega(magnet->system(), 3);

  // 1. If prescribed rigid angular velocity is set, use it
  if (!magnet->rigidAngularVelocity.assuredZero()) {
    int ncells = omega.grid().ncells();
    CuVectorParameter prescribed = magnet->rigidAngularVelocity.cu();
    cudaLaunch(ncells, k_prescribedVectorField, omega.cu(), prescribed);
    return omega;
  }

  // 2. Otherwise compute from elastodynamics
  if (!magnet->enableElastodynamics()) {
    omega.makeZero();
    return omega;
  }

  int ncells = omega.grid().ncells();
  CuField v = magnet->elasticVelocity()->field().cu();
  real3 w = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(ncells, k_rotationVector, omega.cu(), v, w, mastergrid);
  return omega;
}


__global__ void k_magnetoRotationField(CuField hField,
                                       const CuField rotField,
                                       const CuField mField,
                                       const CuVectorParameter anisU,
                                       const CuParameter Kmr,
                                       const CuParameter msat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = hField.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  real3 omega = rotField.vectorAt(idx);
  real3 uhat = anisU.vectorAt(idx);
  real3 m = mField.vectorAt(idx);

  // cross = Omega x uhat
  real3 cross = {omega.y * uhat.z - omega.z * uhat.y,
                 omega.z * uhat.x - omega.x * uhat.z,
                 omega.x * uhat.y - omega.y * uhat.x};

  real a = m.x * cross.x + m.y * cross.y + m.z * cross.z;  // m . (Omega x u)
  real b = m.x * uhat.x + m.y * uhat.y + m.z * uhat.z;    // m . u

  real coeff = Kmr.valueAt(idx) / msat.valueAt(idx);

  hField.setVectorInCell(idx,
    real3{coeff * (a * uhat.x + b * cross.x),
          coeff * (a * uhat.y + b * cross.y),
          coeff * (a * uhat.z + b * cross.z)});
}


Field evalMagnetoRotationField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (magnetoRotationAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  // Get rotation vector from host magnet if sublattice
  Field rot;
  if (magnet->isSublattice()) {
    rot = evalRotationVector(magnet->hostMagnet());
  } else {
    rot = evalRotationVector(magnet);
  }

  int ncells = hField.grid().ncells();
  CuField mField = magnet->magnetization()->field().cu();
  CuVectorParameter anisU = magnet->anisU.cu();
  CuParameter Kmr = magnet->Kmr.cu();
  CuParameter msat = magnet->msat.cu();

  cudaLaunch(ncells, k_magnetoRotationField, hField.cu(), rot.cu(),
             mField, anisU, Kmr, msat);
  return hField;
}


Field evalMagnetoRotationEnergyDensity(const Ferromagnet* magnet) {
  if (magnetoRotationAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalMagnetoRotationField(magnet), 0.5);
}

real evalMagnetoRotationEnergy(const Ferromagnet* magnet) {
  if (magnetoRotationAssuredZero(magnet))
    return 0.0;

  real edens = magnetoRotationEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}


M_FieldQuantity rotationVectorQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalRotationVector, 3,
                         "rotation_vector", "rad/m");
}

M_FieldQuantity angularVelocityQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalAngularVelocity, 3,
                         "angular_velocity", "rad/s");
}

FM_FieldQuantity magnetoRotationFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoRotationField, 3,
                          "magneto_rotation_field", "T");
}

FM_FieldQuantity magnetoRotationEnergyDensityQuantity(
    const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoRotationEnergyDensity, 1,
                          "magneto_rotation_energy_density", "J/m3");
}

FM_ScalarQuantity magnetoRotationEnergyQuantity(
    const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalMagnetoRotationEnergy,
                           "magneto_rotation_energy", "J");
}
