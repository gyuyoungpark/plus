#pragma once

#include "magnet.hpp"
#include "ferromagnet.hpp"
#include "inter_parameter.hpp"
#include <vector>
#include <memory>

class HostMagnet : public Magnet {
public:
    HostMagnet(std::shared_ptr<System> system_ptr, std::string name);

    // Return all sublattices
    virtual std::vector<const Ferromagnet*> sublattices() const;

    void addSublattice(const Ferromagnet* sub);
    std::vector<const Ferromagnet*> getOtherSublattices(const Ferromagnet*) const;

public:
    Parameter afmex_cell;
    Parameter afmex_nn;
    Parameter latcon;
    InterParameter interAfmExchNN;
    InterParameter scaleAfmExchNN;
private:
    std::vector<const Ferromagnet*> sublattices_;
};
