#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "grid.hpp"
#include "voronoi.hpp"
#include "wrappers.hpp"


void wrap_voronoi(py::module& m) {
    py::class_<VoronoiTessellator>(m, "VoronoiTessellator")

        .def(py::init<Grid, real, real3>(), py::arg("grid"), py::arg("grainsize"), py::arg("cellsize"))
        // TODO: create template function (wrap_system.cpp)
        .def("generate", [](VoronoiTessellator t) {
            uint* tess = t.generate().getHostCopy();

            // Create python capsule which will free geometry
            py::capsule free_when_done(tess, [](void* p) {
            uint* tess = reinterpret_cast<uint*>(p);
            delete[] tess;
            });

            int3 size = t.grid.size();
            int shape[3] = {size.z, size.y, size.x};
            int strides[3];
            strides[0] = sizeof(uint) * size.x * size.y;
            strides[1] = sizeof(uint) * size.x;
            strides[2] = sizeof(uint);

            return py::array_t<uint>(shape, strides, tess, free_when_done);
            });
}