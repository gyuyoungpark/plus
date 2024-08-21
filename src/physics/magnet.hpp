#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "strayfield.hpp"
#include "world.hpp"
#include "system.hpp"

class Antiferromagnet;
class Ferromagnet;
class FieldQuantity;
class MumaxWorld;
class System;

class Magnet {
 friend class MumaxWorld;
 public:
  explicit Magnet(std::shared_ptr<System> system_ptr,
                  std::string name);
  explicit Magnet(MumaxWorld* world,
                  Grid grid,
                  std::string name,
                  GpuBuffer<bool> geometry = GpuBuffer<bool>());

  virtual ~Magnet();

  std::string name() const;
  std::shared_ptr<const System> system() const;
  const World* world() const;
  const MumaxWorld* mumaxWorld() const;
  Grid grid() const;
  real3 cellsize() const;
  const GpuBuffer<bool>& getGeometry() const;

  // Cast Magnet instance to child instances
  const Ferromagnet* asFM() const;
  const Antiferromagnet* asAFM() const;

  const StrayField* getStrayField(const Magnet*) const;
  std::vector<const StrayField*> getStrayFields() const;
  void addStrayField(
      const Magnet*,
      StrayFieldExecutor::Method method = StrayFieldExecutor::METHOD_AUTO);
  void removeStrayField(const Magnet*);

 private:
  std::shared_ptr<System> system_;  // the system_ has to be initialized first,
                                    // hence its listed as the first datamember here
  std::string name_;
  std::map<const Magnet*, StrayField*> strayFields_;

 public:
  bool enableAsStrayFieldSource;
  bool enableAsStrayFieldDestination;

  // Delete copy constructor and copy assignment operator to prevent shallow copies
  Magnet(const Magnet&) = delete;
  Magnet& operator=(const Magnet&) = delete;

  Magnet(Magnet&& other) noexcept : system_(other.system_), name_(other.name_) {
        other.system_ = nullptr;
        other.name_ = "";
    }

  // Provide move constructor and move assignment operator
  Magnet& operator=(Magnet&& other) noexcept {
        if (this != &other) {
            system_ = other.system_;
            name_ = other.name_;
            other.system_ = nullptr;
            other.name_ = "";
        }
        return *this;
    }
};