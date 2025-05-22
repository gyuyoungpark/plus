#pragma once

#include "antiferromagnet.hpp"
#include "ncafm.hpp"
#include "quantityevaluator.hpp"

template <typename FuncAFM, typename FuncNCAFM>
FM_FieldQuantity hostExchangeFieldQuantity(const Ferromagnet* magnet, FuncAFM afmFunc, FuncNCAFM ncafmFunc) {
    if (magnet->hostMagnet<Antiferromagnet>()) { return afmFunc(magnet); }
    else if (magnet->hostMagnet<NCAFM>()) { return ncafmFunc(magnet); }
    throw std::runtime_error("Ferromagnet is not part of a recognized host.");
}

template <typename FuncAFM, typename FuncNCAFM>
FM_ScalarQuantity hostExchangeScalarQuantity(const Ferromagnet* magnet, FuncAFM afmFunc, FuncNCAFM ncafmFunc) {
    if (magnet->hostMagnet<Antiferromagnet>()) { return afmFunc(magnet); }
    else if (magnet->hostMagnet<NCAFM>()) { return ncafmFunc(magnet); }
    throw std::runtime_error("Ferromagnet is not part of a recognized host.");
}