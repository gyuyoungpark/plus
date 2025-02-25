#include <math.h>
#include <random>

#include "field.hpp"
#include "gpubuffer.hpp"
#include "voronoi.hpp"

VoronoiTessellator::VoronoiTessellator(real grainsize, unsigned int maxIdx, int seed)
    : grainsize_(grainsize),
      seed_(seed),
      distReal_(0.0, 1.0),
      distInt_(0, maxIdx) {
        real tilesize_in_grains = 2; // tile size in unit grains
        tilesize_ = tilesize_in_grains * grainsize;
        lambda_ = tilesize_in_grains * tilesize_in_grains;
    }

    real VoronoiTessellator::findTileSize(real grid_size, real grainsize) {
        real min_tile_size = 2 * grainsize;
        unsigned int N = std::max(1u, static_cast<unsigned int>(std::ceil(grid_size / min_tile_size)));
        return grid_size / N;  // Adjusted tile size to fit the grid evenly
    }


GpuBuffer<unsigned int> VoronoiTessellator::generate(Grid grid, real3 cellsize) {

    gridsize_ = grid.size();
    cellsize_ = cellsize;

    tilesize_x = findTileSize(grid.size().x*cellsize.x, grainsize_);
    tilesize_y = findTileSize(grid.size().y*cellsize.y, grainsize_);

    numTiles_x = static_cast<int>(std::ceil(grid.size().x * cellsize.x / tilesize_x));
    numTiles_y = static_cast<int>(std::ceil(grid.size().y * cellsize.y / tilesize_y));

    lambda_x = (tilesize_x / grainsize_) * (tilesize_x / grainsize_);
    lambda_y = (tilesize_y / grainsize_) * (tilesize_y / grainsize_);

   lambda_ = std::max(lambda_x, lambda_y);

   std::vector<unsigned int> data(grid.ncells());

   for (int nx = 0; nx < grid.size().x; nx++) {
        for (int ny = 0; ny < grid.size().y; ny++) {
            real3 coo = real3{nx * cellsize.x,
                              ny * cellsize.y,
                              0};
            data[nx + grid.size().x * ny] = regionOf(coo);
        }
    }
    return GpuBuffer<unsigned int>(data);
}

unsigned int VoronoiTessellator::regionOf(real3 coo) {
    Tile t = tileOfCell(coo);
    Center nearest = Center{coo, 0};
    real mindist = INFINITY;
    for (int tx = t.pos.x-1; tx <= t.pos.x+1; tx++) {
        for (int ty = t.pos.y-1; ty <= t.pos.y+1; ty++) {

            int wrapped_tx = (tx + numTiles_x) % numTiles_x;
            int wrapped_ty = (ty + numTiles_y) % numTiles_y;

            std::vector<Center> centers = centersInTile(int3{wrapped_tx, wrapped_ty, 0});

            for (auto c : centers) {
                real3 shifted_center = periodicShift(coo, c.pos, gridsize_.x*cellsize_.x, gridsize_.y*cellsize_.y);
                real dist = (coo.x - shifted_center.x) * (coo.x - shifted_center.x)
                        + (coo.y - shifted_center.y) * (coo.y - shifted_center.y);

                if (dist < mindist) {
                    nearest = c;
                    mindist = dist;
                }
            }
        }
    }
    return nearest.ridx;
}

real3 VoronoiTessellator::periodicShift(real3 coo, real3 center, real grid_size_x, real grid_size_y) {
    real dx = center.x - coo.x;
    real dy = center.y - coo.y;

    // Apply periodic boundary corrections
    if (dx > grid_size_x * 0.5) dx -= grid_size_x;
    if (dx < -grid_size_x * 0.5) dx += grid_size_x;
    if (dy > grid_size_y * 0.5) dy -= grid_size_y;
    if (dy < -grid_size_y * 0.5) dy += grid_size_y;

    return real3{coo.x + dx, coo.y + dy, 0};
}

std::vector<Center> VoronoiTessellator::centersInTile(int3 pos) {

    // Check if centers in this tile are already cached
    auto it = tileCache_.find(pos);
    if (it != tileCache_.end()) {
        return it->second.centers;
    }

    int64_t seed = (int64_t(pos.y) + (1<<24)) * (1<<24)
                 + (int64_t(pos.x) + (1<<24));
    engine_.seed (seed ^ seed_);

    int N = Poisson(lambda_);

    std::vector<Center> centers(N);

    for (int n = 0; n < N; n++) {
        real cx = (pos.x + distReal_(engine_)) * tilesize_x;
        real cy = (pos.y + distReal_(engine_)) * tilesize_y;

        centers[n] = Center(real3{cx, cy, 0}, distInt_(engine_));
    }
    
    // Cache centers belonging to this tile
    Tile newTile;
    newTile.pos = pos;
    newTile.centers = centers;
    tileCache_[pos] = newTile;

    return centers;
}

int VoronoiTessellator::Poisson(real lambda) {
    std::poisson_distribution<int> dist(lambda);
    return dist(engine_);
}

Tile VoronoiTessellator::tileOfCell(real3 coo) {
    return Tile{int3{static_cast<int>(std::floor(coo.x / tilesize_x)), // This cannot be the best way...
                     static_cast<int>(std::floor(coo.y / tilesize_y)),
                     0}
            };
}