#ifndef QUADRATURE1D_HPP
#define QUADRATURE1D_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

namespace heat2d::quadrature {

class Quadrature1D {
 public:
  virtual ~Quadrature1D() = default;
  virtual void compute(int numPoints, Eigen::VectorXd& nodes,
                        Eigen::VectorXd& weights) const = 0;
};

// Gauss-Legendre quadrature on the domain [-1,1]. This rule is exact for polynomials with degree at most 2n-1.
class GaussLegendre : public Quadrature1D {
 public:
  void compute(int numPoints, Eigen::VectorXd& nodes,
               Eigen::VectorXd& weights) const override {
    if (numPoints < 1) throw std::invalid_argument("numPoints must be >= 1");

    // Jacobi matrix: symmetric tridiagonal, zero diagonal,
    // off-diagonal beta_k = k / sqrt(4k^2 - 1), k = 1..n-1
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(numPoints, numPoints);
    for (int k = 1; k < numPoints; ++k) {
      double beta = k / std::sqrt(4.0 * k * k - 1.0);
      J(k, k - 1) = beta;
      J(k - 1, k) = beta;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(J);
    nodes = solver.eigenvalues();
    // weight_i = 2 * (first component of i-th eigenvector)^2
    weights = 2.0 * solver.eigenvectors().row(0).array().square();
  }
};

}  // namespace heat2d::quadrature

#endif  // ifndef QUADRATURE1D_HPP