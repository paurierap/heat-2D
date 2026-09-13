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
  mutable Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> SPDsolver_;
  mutable Eigen::SparseLU<Eigen::SparseMatrix<double>> LUsolver_;

  bool isMatrixSPD_ = false;

 public:
  ExplicitEuler(double timestep) : TimeIntegrator(timestep) {};

  void setUp(const solver::SpatialDiscretization2D& sd) override {
    isMatrixSPD_ = sd.isMSPD();

    if (isMatrixSPD_) {
      SPDsolver_.compute(sd.getMatrixM());
      if (SPDsolver_.info() != Eigen::Success)
        throw std::runtime_error(
            "LDLT factorization for Explicit Euler mass matrix failed\n");
    } else {
      LUsolver_.compute(sd.getMatrixM());
      if (LUsolver_.info() != Eigen::Success)
        throw std::runtime_error(
            "LU factorization for Explicit Euler mass matrix failed\n");
    }
  };

  void step(solver::SpatialDiscretization2D& sd, double t,
            Eigen::VectorXd& u) const override {
    sd.updateRHS(t);

    const Eigen::SparseMatrix<double>& K = sd.getMatrixK();
    const Eigen::VectorXd& b = sd.getVector();

    Eigen::VectorXd rhs = K * u + b;

    if (isMatrixSPD_) {
      u += (timestep_ * SPDsolver_.solve(rhs)).eval();
    } else {
      u += (timestep_ * LUsolver_.solve(rhs)).eval();
    }
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
