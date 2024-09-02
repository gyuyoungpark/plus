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

class VoronoiTesselator {
 public:
  VoronoiTesselator(Grid grid, real grainsize, real3 cellsize);
  ~VoronoiTesselator() = default;

  // * Generate a Voronoi tesselation
  Field generate();

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
  Grid grid_;
  real grainsize_;
  real3 cellsize_;
  real tilesize_;
  std::unordered_map<int3, std::vector<Center>, Int3Hash> centerCache_;

 private:
 // RNG related members
  real lambda_; // Poisson parameter
  std::default_random_engine engine_;
  std::uniform_real_distribution<> distReal_;
  std::uniform_int_distribution<uint> distInt_; 
};