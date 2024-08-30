#include <random>
#include <vector>

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
 private:
 // RNG related members
  real lambda_; // Poisson parameter
  std::default_random_engine engine_;
  std::uniform_real_distribution<> distReal_;
  std::uniform_int_distribution<uint> distInt_; 

/** IS THIS NECESSARY?????
 * 
 */
  friend Center;
};

/*
- Generate tiles
  * each tile has an index (crf. gpubuffer of regions)

- per tile:
  * calc # centers using Poisson

- for each cell, look at centers in NN tiles to see to which region it belongs
  * uiteindelijk dan:
    for cell in grid:
      reg_idx = regionOf(cell)
  
    mss zo index-field maken?, dan fieldToArray naar Python
  
  
  */