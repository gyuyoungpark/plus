#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "grid.hpp"
#include "voronoi.hpp"
#include "wrappers.hpp"


void wrap_voronoi(py::module& m) {
    py::class_<VoronoiTessellator>(m, "VoronoiTessellator")

        .def(py::init<Grid, real, real3>(), py::arg("grid"), py::arg("grainsize"), py::arg("cellsize"))
        .def("generate", [](VoronoiTessellator t) { return fieldToArray<uint>(t.generate()); });
}