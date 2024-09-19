#include <memory>
#include <string>
#include <vector>

#include "butchertableau.hpp"
#include "dynamicequation.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "timesolver.hpp"
#include "variable.hpp"
#include "wrappers.hpp"
#include <pybind11/functional.h>  // for run_while

void wrap_timesolver(py::module& m) {
  py::class_<TimeSolver>(m, "TimeSolver")
      .def_property("time", &TimeSolver::time, &TimeSolver::setTime)
      .def_property_readonly("sensible_timestep", &TimeSolver::sensibleTimeStep)
      .def("set_method",
           [](TimeSolver& solver, std::string methodName) {
             RKmethod method = getRungeKuttaMethodFromName(methodName);
             solver.setRungeKuttaMethod(method);
           })
      .def_property("max_error", &TimeSolver::maxerror, &TimeSolver::setMaxError)
      .def("step", &TimeSolver::step)
      .def("steps", &TimeSolver::steps)
      .def_property("timestep", &TimeSolver::timestep, &TimeSolver::setTimeStep)
      .def_property("adaptive_timestep", &TimeSolver::hasAdaptiveTimeStep,
                    [](TimeSolver& solver, bool adaptive) {
                      if (adaptive) {
                        solver.enableAdaptiveTimeStep();
                      } else {
                        solver.disableAdaptiveTimeStep();
                      }
                    })
      .def("run", &TimeSolver::run)
      .def("run_while", &TimeSolver::runwhile);
}
