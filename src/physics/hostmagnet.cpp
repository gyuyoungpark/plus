#include "hostmagnet.hpp"

HostMagnet::HostMagnet(std::shared_ptr<System> system_ptr, std::string name)
    : Magnet(system_ptr, name) {}

std::vector<const Ferromagnet*> HostMagnet::sublattices() const {
    return sublattices_;
}

void HostMagnet::addSublattice(const Ferromagnet* sub) {
    sublattices_.push_back(sub);
}
