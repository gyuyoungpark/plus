#pragma once

#include "magnet.hpp"
#include "ferromagnet.hpp"
#include <vector>
#include <memory>

class HostMagnet : public Magnet {
public:
    HostMagnet(std::shared_ptr<System> system_ptr, std::string name);

    // Return all sublattices
    virtual std::vector<const Ferromagnet*> sublattices() const;

    void addSublattice(const Ferromagnet* sub);

private:
    std::vector<const Ferromagnet*> sublattices_;
};
