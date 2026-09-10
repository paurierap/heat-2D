#ifndef QUADRATURE2D_HPP
#define QUADRATURE2D_HPP
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

#include "Quadrature1D.hpp"
#include "triangle_dunavant_rule.hpp"

namespace heat2d::quadrature {

class Quadrature2D {
 protected:
  Eigen::MatrixX2d points_;
  Eigen::VectorXd weights_;

 public:
  virtual ~Quadrature2D() = default;

  virtual void compute(int numPoints) = 0;

  const Eigen::MatrixX2d& getPoints() const { return points_; }
  const Eigen::VectorXd& getWeights() const { return weights_; }
  int numPoints() const { return static_cast<int>(weights_.size()); }
};

// Dunavant quadrature rules for the reference triangle with vertices (0,0),
// (1,0), and (0,1). This rule is exact for polynomials of total degree at most
// d, where d depends on the rule number. Wrapper for Dunavant quadrature rules
// for triangles, extracted from
// <https://people.math.sc.edu/burkardt/cpp_src/triangle_dunavant_rule/triangle_dunavant_rule.html/>.
class TriangleQuadrature : public Quadrature2D {
 public:
  void compute(int rule) override {
    if (rule < 1 || rule > dunavant_rule_num())
      throw std::invalid_argument("Dunavant: unsupported rule.");

    int n = dunavant_order_num(rule);
    std::vector<double> xy(2 * n), w(n);
    dunavant_rule(rule, n, xy.data(), w.data());

    points_.resize(n, 2);
    weights_.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
      points_(i, 0) = xy[2 * i];
      points_(i, 1) = xy[2 * i + 1];
      weights_(i) = w[i];
    }
  }
};

// Tensor-product Gauss-Legendre quadrature on the reference square [-1,1] x
// [-1,1]. This rule is exact for polynomials of total separable degree at most
// 2n-1 in each coordinate.
class QuadQuadrature : public Quadrature2D {
 public:
  void compute(int n) override {
    if (n < 1) throw std::invalid_argument("QuadQuadrature: n must be >= 1");

    GaussLegendre rule;
    rule.compute(n);

    points_.resize(n * n, 2);
    weights_.resize(n * n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        points_(i * n + j, 0) = rule.getNodes()(i);
        points_(i * n + j, 1) = rule.getNodes()(j);
        weights_(i * n + j) = rule.getWeights()(i) * rule.getWeights()(j);
      }
    }
  }
};

}  // namespace heat2d::quadrature

#endif  // QUADRATURE2D_HPP