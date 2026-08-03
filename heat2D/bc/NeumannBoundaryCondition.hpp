#ifndef NEUMANNBOUNDARYCONDITION_HPP
#define NEUMANNBOUNDARYCONDITION_HPP

#include "BoundaryCondition.hpp"

namespace heat2d::bc {

class NeumannBoundaryCondition : public BoundaryCondition {
 public:
  NeumannBoundaryCondition(std::function<double(double, double, double)> f)
      : BoundaryCondition(f) {};

  BoundaryConditionType getType() const override {
    return BoundaryConditionType::Neumann;
  };
};

}  // namespace heat2d::bc
#endif  // ifndef NEUMANNBOUNDARYCONDITION_HPP