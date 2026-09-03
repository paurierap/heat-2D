#ifndef TRIANGLE_QUADRATURE_HPP
#define TRIANGLE_QUADRATURE_HPP

// Wrapper for Dunavant quadrature rules for triangles, extracted from <https://people.math.sc.edu/burkardt/cpp_src/triangle_dunavant_rule/triangle_dunavant_rule.html/>.

#include <Eigen/Dense>
#include <stdexcept>

#include "Quadrature2D.hpp"
#include "triangle_dunavant_rule.hpp"

namespace heat2d::quadrature {

// Dunavant quadrature rules for the reference triangle with vertices (0,0), (1,0), and (0,1). This rule is exact for polynomials of total degree at most d, where d depends on the rule number.
class TriangleQuadrature : public Quadrature2D {
 public:
  explicit TriangleQuadrature(int rule) : Quadrature2D() {
    if (rule < 1 || rule > dunavant_rule_num())
      throw std::invalid_argument("Dunavant: unsupported rule.");

    int n = dunavant_order_num(rule);
    std::vector<double> xy(2 * n), w(n);
    dunavant_rule(rule, n, xy.data(), w.data());

    points_.resize(n, 2);
    weights_.resize(n);
    for (u_int i = 0; i < n; ++i) {
      points_(i, 0) = xy[2 * i];
      points_(i, 1) = xy[2 * i + 1];
      weights_(i) = w[i];
    }
  }
};

}  // namespace heat2d::quadrature

#endif  // TRIANGLE_QUADRATURE_HPP