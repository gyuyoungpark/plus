#include <math.h>
#include <random>

#include "field.hpp"
#include "voronoi.hpp"

VoronoiTesselator::VoronoiTesselator(Grid grid, real grainsize, real3 cellsize)
    : grid_(grid),
      grainsize_(grainsize),
      cellsize_(cellsize),
      distReal_(0.0, 1.0),
      distInt_(std::numeric_limits<uint>::min(),
               std::numeric_limits<uint>::max()) {
        real tilesize_in_grains = 2; // tile size in unit grains
        tilesize_ = tilesize_in_grains * grainsize;
        lambda_ = tilesize_in_grains * tilesize_in_grains;
    }

Field VoronoiTesselator::generate() {

   std::vector<real> data;
   for (int nx = 0; nx < grid_.size().x; nx++) {
        for (int ny = 0; ny < grid_.size().y; ny++) {
            for (int nz = 0; nz < grid_.size().z; nz++) {
                real3 coo = real3{nx * cellsize_.x,
                                  ny * cellsize_.y,
                                  0};
                uint ridx = regionOf(coo);
                data.push_back(ridx);
            }
        }
    }
    Field idxField(1, grid_.size(), data.size());
    idxField.setData(data);

    return idxField;
}

uint VoronoiTesselator::regionOf(real3 coo) {
    Tile t = tileOfCell(coo);
    Center nearest = Center{coo, 0};
    real mindist = INFINITY;
    for (int tx = t.pos.x-1; tx <= t.pos.x+1; tx++) {
        for (int ty = t.pos.y-1; ty <= t.pos.y+1; ty++) {
            std::vector<Center> centers = centersInTile(int3{tx, ty, 0});
            for (auto c : centers) {
                real dist = (coo.x - c.pos.x) * (coo.x - c.pos.x)
                          + (coo.y - c.pos.y) * (coo.y - c.pos.y);
                if (dist < mindist) {
                    nearest = c;
                    mindist = dist;
                }
            }
        }
    }
    return nearest.ridx;
}

std::vector<Center> VoronoiTesselator::centersInTile(int3 pos) {
// TODO: cache centers so not to calculate this multiple times.
    int64_t seed = (int64_t(pos.y) + (1<<24)) * (1<<24)
                 + (int64_t(pos.x) + (1<<24));
    engine_.seed (seed ^ 1234567);

    int N = Poisson(lambda_);

    std::vector<Center> centers;

    for (int n = 0; n < N; n++) {
        Center c;
        real cx = (pos.x + distReal_(engine_)) * tilesize_;
        real cy = (pos.y + distReal_(engine_)) * tilesize_;

        c.pos = real3{cx, cy, 0};
        c.ridx = distInt_(engine_);
        centers.push_back(c);

    }
    return centers;
}

int VoronoiTesselator::Poisson(real lambda) {
    std::poisson_distribution<int> dist(lambda);
    return dist(engine_);
}

Tile VoronoiTesselator::tileOfCell(real3 coo) {
    return Tile{int3{static_cast<int>(std::floor(coo.x / tilesize_)), // This cannot be the best way...
                     static_cast<int>(std::floor(coo.y / tilesize_)),
                     0}
            };
}

