#include <memory>
#include <stdexcept>

#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "strayfieldkernel.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_magnet(py::module& m) {
  py::class_<Magnet>(m, "Magnet")
      .def_property_readonly("name", &Magnet::name)
      .def_property_readonly("system", &Magnet::system)
      .def_property_readonly("world", &Magnet::mumaxWorld)

      .def_readwrite("enable_as_stray_field_source",
                     &Magnet::enableAsStrayFieldSource)
      .def_readwrite("enable_as_stray_field_destination",
                     &Magnet::enableAsStrayFieldDestination)

      .def("stray_field_from_magnet",
          [](const Magnet* m, Magnet* magnet) {
            const StrayField* strayField = m->getStrayField(magnet);
            if (!strayField)
              throw std::runtime_error(
                  "Can not compute the magnetic field of the magnet");
            return strayField;
          },
          py::return_value_policy::reference)
  ;

  m.def("_demag_kernel", [](const Magnet* m) {
    StrayFieldKernel demagKernel(m->grid(), m->grid(), m->world());
    return fieldToArray(demagKernel.field());
  });
}
