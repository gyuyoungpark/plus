#include <random>
#include <vector>
#include <unordered_map>

#include "datatypes.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"

struct Center {
  real3 pos;
  uint ridx;
  Center() : pos{0, 0, 0}, ridx(0) {}
  Center(real3 position, uint region_idx)
      : pos(position), ridx(region_idx) {}
};

struct Tile {
  int3 pos;
  std::vector<Center> centers;
};

struct Int3Hash {
  // Hash function to allow int3 to be used as a key
    std::size_t operator()(const int3& k) const {
        return std::hash<int>()(k.x) ^ std::hash<int>()(k.y) ^ std::hash<int>()(k.z);
    }
};

class VoronoiTessellator {
 public:
  VoronoiTessellator(Grid grid, real grainsize, real3 cellsize);
  ~VoronoiTessellator() = default;

  // * Generate a Voronoi tessellation
  GpuBuffer<uint> generate();

 private:
  // * Calc nearest center and assign center index to coo
  uint regionOf(real3 coo);

  // * Calculate position and index of centers in tile
  std::vector<Center> centersInTile(int3 pos);

  // * Poisson distribution
  int Poisson(real lambda);
  
  // * Calculate to which tile the given cell belongs
  Tile tileOfCell(real3 coo);

public:
  Grid grid;
private:
  real grainsize_;
  real3 cellsize_;
  real tilesize_;
  uint centerIdx_ = 0;
  std::unordered_map<int3, Tile, Int3Hash> tileCache_;

 // RNG related members
  real lambda_; // Poisson parameter
  std::default_random_engine engine_;
  std::uniform_real_distribution<> distReal_;
};