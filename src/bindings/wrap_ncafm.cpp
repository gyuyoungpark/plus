#include <memory>
#include <stdexcept>

#include "ncafm.hpp"
#include "energy.hpp"
#include "fieldquantity.hpp"
#include "fullmag.hpp"
#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "ncafmexchange.hpp"
#include "octupole.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_ncafm(py::module& m) {
  py::class_<NCAFM, Magnet>(m, "NCAFM")
      .def("sub1", &NCAFM::sub1, py::return_value_policy::reference)
      .def("sub2", &NCAFM::sub2, py::return_value_policy::reference)
      .def("sub3", &NCAFM::sub3, py::return_value_policy::reference)
      .def("sublattices", &NCAFM::sublattices, py::return_value_policy::reference)

      .def_readonly("ncafmex_cell", &NCAFM::afmex_cell)
      .def_readonly("ncafmex_nn", &NCAFM::afmex_nn)
      .def_readonly("inter_ncafmex_nn", &NCAFM::interAfmExchNN)
      .def_readonly("scale_ncafmex_nn", &NCAFM::scaleAfmExchNN)
      .def_readonly("latcon", &NCAFM::latcon)
      .def_readonly("dmi_tensor", &NCAFM::dmiTensor)
      .def_readonly("dmi_vector", &NCAFM::dmiVector)

      .def("minimize", &NCAFM::minimize, py::arg("tol"), py::arg("nsamples"))
      .def("relax", &NCAFM::relax, py::arg("tol"));

  m.def("octupole_vector", &octupoleVectorQuantity);
  m.def("full_magnetization",
        py::overload_cast<const NCAFM*>(&fullMagnetizationQuantity));

  m.def("angle_field", &angleFieldQuantity);
  m.def("max_intracell_angle_between",
          [](const Ferromagnet* i, const Ferromagnet* j) { return evalMaxAngle(i, j); },
          py::arg("sub1"), py::arg("sub2"));

  m.def("total_energy_density",
        py::overload_cast<const NCAFM*>(&totalEnergyDensityQuantity));
  m.def("total_energy",
        py::overload_cast<const NCAFM*>(&totalEnergyQuantity));
}