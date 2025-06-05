#include "traction.hpp"

#include "parameter.hpp"
#include "system.hpp"

BoundaryTraction::BoundaryTraction(std::shared_ptr<const System> system)
    : posXside(system),
      negXside(system),
      posYside(system),
      negYside(system),
      posZside(system),
      negZside(system) {}

CuBoundaryTraction BoundaryTraction::cu() const {
  return CuBoundaryTraction{
      posXside.cu(), negXside.cu(), 
      posYside.cu(), negYside.cu(), 
      posZside.cu(), negZside.cu()
  };
}

bool BoundaryTraction::assuredZero() const {
  return posXside.assuredZero() && negXside.assuredZero() &&
         posYside.assuredZero() && negYside.assuredZero() &&
         posZside.assuredZero() && negZside.assuredZero();
}
