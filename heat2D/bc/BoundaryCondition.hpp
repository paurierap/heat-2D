#ifndef BOUNDARYCONDITION_HPP
#define BOUNDARYCONDITION_HPP

#include <functional>
#include <vector>

namespace heat2d::bc {

enum class BoundaryConditionType { Dirichlet, Neumann, Robin };

class BoundaryCondition {
 protected:
  std::function<double(double, double, double)> f_;

 public:
  BoundaryCondition(std::function<double(double, double, double)> f) : f_(f) {};
  virtual ~BoundaryCondition() = default;

  virtual BoundaryConditionType getType() const = 0;
  inline double f(double x, double y, double t = 0.0) const {
    return f_(x, y, t);
  };

  // General ghost-node relation: u_coeff*u + du_coeff*du/dn = f.
  // Defaults correspond to Neumann (u_coeff=0, du_coeff=1); Dirichlet never
  // calls these (handled by node elimination), Robin overrides both.
  virtual double u_coeff(double, double, double = 0.0) const { return 0.0; }
  virtual double du_coeff(double, double, double = 0.0) const { return 1.0; }
};

}  // namespace heat2d::bc
#endif  // ifndef BOUNDARYCONDITION_HPP