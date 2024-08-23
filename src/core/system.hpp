#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "world.hpp"

class CuSystem;

class System {
 public:
  /** Construct a system with a given grid which lives in a given world. */
  System(const World* world,
         Grid grid,
         GpuBuffer<bool> geometry = GpuBuffer<bool>(),
         GpuBuffer<uint> regions = GpuBuffer<uint>());

  // Systems should not be copied or moved
  System(const System&) = delete;
  System& operator=(const System&) = delete;
  System(System&&) = delete;
  System& operator=(System&&) = delete;

  /** Destroy the system. */
  virtual ~System() {}

  /** Return the world to which the system belongs. */
  const World* world() const;

  /** Return the grid of the system. */
  Grid grid() const;

  /** Return the cellsize of the world to which the system belongs. */
  real3 cellsize() const;

  /** Return the position of a cell of this system in the world. */
  real3 cellPosition(int3) const;

  /** Return the position of the origin of this system in the world. */
  real3 origin() const;

  /** Return the position of the center of this system in the world. */
  real3 center() const;

  /** Get the geometry of the system. */
  const GpuBuffer<bool>& geometry() const;

  /** Get the regions of the system. */
  const GpuBuffer<uint>& regions() const;

  /** Return the number of cells which lie within the geometry*/
  int cellsingeo() const;

  /** Return a CuSystem which can be copied to the gpu and be used in cuda
   * kernels. */
  CuSystem cu() const;

 private:
  const World* world_;
  Grid grid_;
  GpuBuffer<bool> geometry_;
  GpuBuffer<uint> regions_;

  friend CuSystem;
};

struct CuSystem {
  explicit CuSystem(const System*);

  const Grid grid;
  const real3 cellsize;
  bool const* geometry = nullptr;
  uint const* regions = nullptr;

  __device__ uint getRegionIdx(int3 coo) const;
  __device__ uint getRegionIdx(int idx) const;

  __device__ bool inGeometry(int3 coo) const;
  __device__ bool inGeometry(int idx) const;
  __device__ bool inRegion(uint regionIdx, int3 coo) const;
  __device__ bool inRegion(uint regionIdx, int idx) const;
};

__device__ inline uint CuSystem::getRegionIdx(int3 coo) const {
  if (!regions) { return 0; }
  else { return regions[grid.coord2index(coo)]; }
}

__device__ inline uint CuSystem::getRegionIdx(int idx) const {
  if (!regions) { return 0; }
  else { return regions[idx]; }
}

__device__ inline bool CuSystem::inGeometry(int3 coo) const {
  return grid.cellInGrid(coo) && (!geometry || geometry[grid.coord2index(coo)]);
}

__device__ inline bool CuSystem::inGeometry(int idx) const {
  return grid.cellInGrid(idx) && (!geometry || geometry[idx]);
}

__device__ inline bool CuSystem::inRegion(uint regionIdx, int3 coo) const {
  return grid.cellInGrid(coo) && (!regions || regionIdx == getRegionIdx(coo));
}

__device__ inline bool CuSystem::inRegion(uint regionIdx, int idx) const {
  return grid.cellInGrid(idx) && (!regions || regionIdx == getRegionIdx(idx));
}