#ifndef ROBINBOUNDARYCONDITION_HPP
#define ROBINBOUNDARYCONDITION_HPP

#include "BoundaryCondition.hpp"

namespace heat2d::bc {

// Robin boundary condition, generally of the form: u_coeff * u + du_coeff *
// du/dn = f, where u_coeff and du_coeff are coefficients that can vary along
// the boundary. The function f(x,y,t) represents the boundary condition value
// at a given point (x,y) and time t.
class RobinBoundaryCondition : public BoundaryCondition {
 private:
  std::function<double(double, double, double)> u_coeff_,
      du_coeff_;  // Coefficients for the Robin boundary condition

 public:
  RobinBoundaryCondition(std::function<double(double, double, double)> u_coeff,
                         std::function<double(double, double, double)> du_coeff,
                         std::function<double(double, double, double)> f)
      : BoundaryCondition(f), u_coeff_(u_coeff), du_coeff_(du_coeff) {};

  BoundaryConditionType getType() const override {
    return BoundaryConditionType::Robin;
  };

  double u_coeff(double x, double y, double t = 0.0) const override {
    return u_coeff_(x, y, t);
  };
  double du_coeff(double x, double y, double t = 0.0) const override {
    return du_coeff_(x, y, t);
  };
};

}  // namespace heat2d::bc
#endif  // ifndef ROBINBOUNDARYCONDITION_HPP