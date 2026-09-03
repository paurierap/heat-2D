#ifndef QUAD_QUADRATURE_HPP
#define QUAD_QUADRATURE_HPP

#include <Eigen/Dense>
#include <stdexcept>

#include "Quadrature1D.hpp"
#include "Quadrature2D.hpp"

namespace heat2d::quadrature {

// Tensor-product Gauss-Legendre quadrature on the reference square [-1,1] x
// [-1,1]. This rule is exact for polynomials of total separable degree at most 2n-1 in each coordinate.
class QuadQuadrature : public Quadrature2D {
 public:
  explicit QuadQuadrature(int n) : Quadrature2D() {
    if (n < 1) throw std::invalid_argument("QuadQuadrature: n must be >= 1");

    Eigen::VectorXd nodes, weights;
    GaussLegendre rule;
    rule.compute(n, nodes, weights);

    points_.resize(n * n, 2);
    weights_.resize(n * n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        points_(i * n + j, 0) = nodes(i);
        points_(i * n + j, 1) = nodes(j);
        weights_(i * n + j) = weights(i) * weights(j);
      }
    }
  }
};

}  // namespace heat2d::quadrature

#endif  // QUAD_QUADRATURE_HPP
