#ifndef IMPLICITEULER_HPP
#define IMPLICITEULER_HPP

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <cassert>
#include <iostream>
#include <memory>

#include "TimeIntegrator.hpp"

namespace heat2d::ode {

class ImplicitEuler : public TimeIntegrator {
 private:
  Eigen::SparseMatrix<double> M_rhs_;
  Eigen::SparseMatrix<double> M_lhs_;

  // Direct solvers
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> SPDsolver_;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> LUsolver_;

  // Iterative solvers
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                           Eigen::Lower | Eigen::Upper>
      CGsolver_;
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> BiCGSTABsolver_;

  // Buffer vectors
  mutable Eigen::VectorXd tmp_;

  bool useIterativeSolver_;
  bool isInitialized_ = false;
  bool isMatrixSPD_ = false;

 public:
  ImplicitEuler(double timestep) : TimeIntegrator(timestep) {};

  void setUp(const solver::SpatialDiscretization2D& sd) override {
    const Eigen::SparseMatrix<double>& M = sd.getMatrixM();
    const Eigen::SparseMatrix<double>& K = sd.getMatrixK();

    // Heuristic for iterative solver choice
    useIterativeSolver_ = (timestep_ * M.rows() < 200.);

    tmp_.resize(M.rows());

    M_rhs_ = M;
    M_lhs_ = M - timestep_ * K;

    isMatrixSPD_ = sd.isMSPD() && sd.isKSPD();

    if (isMatrixSPD_) {
      if (useIterativeSolver_) {
        CGsolver_.setMaxIterations(1000);
        CGsolver_.setTolerance(1e-10);
        CGsolver_.compute(M_lhs_);
        if (CGsolver_.info() != Eigen::Success)
          throw std::runtime_error(
              "Conjugate Gradient factorization for Implicit Euler failed\n");
      } else {
        SPDsolver_.compute(M_lhs_);
        if (SPDsolver_.info() != Eigen::Success)
          throw std::runtime_error(
              "LDLT factorization for Implicit Euler failed\n");
      }
    } else {
      if (useIterativeSolver_) {
        BiCGSTABsolver_.setMaxIterations(1000);
        BiCGSTABsolver_.setTolerance(1e-10);
        BiCGSTABsolver_.compute(M_lhs_);
        if (BiCGSTABsolver_.info() != Eigen::Success)
          throw std::runtime_error(
              "BiCGSTAB factorization for Implicit Euler failed\n");
      } else {
        LUsolver_.compute(M_lhs_);
        if (LUsolver_.info() != Eigen::Success)
          throw std::runtime_error(
              "LU factorization for Implicit Euler failed\n");
      }
    }

    isInitialized_ = true;
  }

  void step(solver::SpatialDiscretization2D& sd, double t,
            Eigen::VectorXd& u) const override {
    if (!isInitialized_)
      throw std::logic_error(
          "\nStep function for Implicit Euler time integration was used before "
          "SetUp.\n");

    sd.updateRHS(t + timestep_);
    const Eigen::VectorXd& b = sd.getVector();

    tmp_.noalias() = M_rhs_ * u;
    tmp_ += timestep_ * b;

    if (isMatrixSPD_) {
      if (useIterativeSolver_) {
        u = CGsolver_.solveWithGuess(tmp_, u);
      } else {
        u = SPDsolver_.solve(tmp_);
      }
    } else {
      if (useIterativeSolver_) {
        u = BiCGSTABsolver_.solveWithGuess(tmp_, u);
      } else {
        u = LUsolver_.solve(tmp_);
      }
    }
  };

  // Virtual factory for timestep remainder operations. Note that the clone does
  // not transfer precomputed matrices. The caller must invoke setUp() on the
  // clone.
  std::unique_ptr<TimeIntegrator> cloneWithTimestep(
      double timestep) const override {
    return std::make_unique<ImplicitEuler>(timestep);
  }
};

}  // namespace heat2d::ode
#endif  // ifndef
