#include "hostmagnet.hpp"

HostMagnet::HostMagnet(std::shared_ptr<System> system_ptr, std::string name)
    : Magnet(system_ptr, name),
      afmex_cell(system(), 0.0, name + ":afmex_cell", "J/m"),
      afmex_nn(system(), 0.0, name + ":afmex_nn", "J/m"),
      latcon(system(), 0.35e-9, name + ":latcon", "m"),
      interAfmExchNN(system(), 0.0, name + ":inter_afmex_nn", "J/m"),
      scaleAfmExchNN(system(), 1.0, name + ":scale_afmex_nn", "") {}

std::vector<const Ferromagnet*> HostMagnet::sublattices() const {
    return sublattices_;
}

void HostMagnet::addSublattice(const Ferromagnet* sub) {
    sublattices_.push_back(sub);
}

std::vector<const Ferromagnet*> HostMagnet::getOtherSublattices(const Ferromagnet* sub) const {
    if (std::find(sublattices_.begin(), sublattices_.end(), sub) == sublattices_.end())
        throw std::out_of_range("Sublattice not found in HostMagnet.");
    std::vector<const Ferromagnet*> result;
    for (auto s : sublattices_) {
        if (s != sub)
            result.push_back(s);
    }
    return result;
}