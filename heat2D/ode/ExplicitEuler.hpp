#ifndef EXPLICITEULER_HPP
#define EXPLICITEULER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "TimeIntegrator.hpp"

namespace heat2d::ode {

class ExplicitEuler : public TimeIntegrator {
 private:
  // TO DO: Consider lumping the mass matrix from FEM.
  mutable Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Msolver_;

 public:
  ExplicitEuler(double timestep) : TimeIntegrator(timestep) {};

  void setUp(const solver::SpatialDiscretization2D& sd) override {
    Msolver_.compute(sd.getMatrixM());
  };

  void step(solver::SpatialDiscretization2D& sd, double t,
            Eigen::VectorXd& u) const override {
    sd.updateRHS(t);

    const Eigen::SparseMatrix<double>& K = sd.getMatrixK();
    const Eigen::VectorXd& b = sd.getVector();

    // Prevent aliasing from expression templating in Eigen using eval()
    u += (timestep_ * Msolver_.solve(K * u + b)).eval();
  };

  // Virtual factory for timestep remainder operations. Note that the clone does
  // not transfer precomputed matrices. The caller must invoke setUp() on the
  // clone.
  std::unique_ptr<TimeIntegrator> cloneWithTimestep(
      double timestep) const override {
    return std::make_unique<ExplicitEuler>(timestep);
  }
};

}  // namespace heat2d::ode
#endif  // ifndef
