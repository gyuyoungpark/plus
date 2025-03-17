#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fullmag.hpp"
#include "ncafm.hpp"

Field evalFullMag(const Ferromagnet* magnet) {
  return magnet->msat.eval() * magnet->magnetization()->field();
}

Field evalAFMFullMag(const Antiferromagnet* magnet) {
  return add(magnet->sub1()->msat.eval(),
             magnet->sub1()->magnetization()->field(),
             magnet->sub2()->msat.eval(),
             magnet->sub2()->magnetization()->field());
}

Field evalNCAFMFullMag(const NCAFM* magnet) {
  return add(magnet->sub1()->msat.eval(),
             magnet->sub1()->magnetization()->field(),
             magnet->sub2()->msat.eval(),
             magnet->sub2()->magnetization()->field(),
             magnet->sub3()->msat.eval(),
             magnet->sub3()->magnetization()->field());
}

FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet* magnet) {
    return FM_FieldQuantity(magnet, evalFullMag, 3, "full_magnetization", "A/m");
}

AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet* magnet) {
    return AFM_FieldQuantity(magnet, evalAFMFullMag, 3, "full_magnetization", "A/m");
}

NCAFM_FieldQuantity fullMagnetizationQuantity(const NCAFM* magnet) {
    return NCAFM_FieldQuantity(magnet, evalNCAFMFullMag, 3, "full_magnetization", "A/m");
}