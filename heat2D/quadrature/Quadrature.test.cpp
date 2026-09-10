#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "Quadrature1D.hpp"
#include "Quadrature2D.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace heat2d::quadrature;

// =============================================================================
// Quadrature1D tests
// =============================================================================
class Quadrature1DTest : public testing::Test {
 protected:
  GaussLegendre q;
  static constexpr double tol = 1e-9;
};

// Closed-form integral of x^k over [-1, 1].
double gauss_exact_solution(int a) {
  return (1.0 - std::pow(-1.0, a + 1)) / (a + 1.0);
}

// =============================================================================
// Test 1 - Integration of polynomials up to degree 2n-1
// =============================================================================

TEST_F(Quadrature1DTest, IntegratesPolynomialsUpToDegree2nMinus1) {
  for (int n = 1; n <= 6; ++n) {
    q.compute(n);

    for (int degree = 0; degree <= 2 * n - 1; ++degree) {
      double approx = 0.0;
      for (int i = 0; i < n; ++i)
        approx += q.getWeights()(i) * std::pow(q.getNodes()(i), degree);
      EXPECT_NEAR(approx, gauss_exact_solution(degree), tol)
          << "n = " << n << ", degree = " << degree;
    }
  }
}

// =============================================================================
// Test 2 - Integration of generic function
// Verify Gauss-Legendre integration of f(x) = exp(x) * cos(pi * x / 2) over
// [-1, 1]. Exact integral: 2π(1+e^2) / (e(π^2 + 4)).
// =============================================================================

TEST_F(Quadrature1DTest, IntegratesGenericFunction) {
  const double exact =
      2 * M_PI * (1.0 + std::exp(2.0)) / (std::exp(1.0) * (M_PI * M_PI + 4.0));

  for (int n = 5; n <= 10; ++n) {
    q.compute(n);

    double approx = 0.0;
    for (int i = 0; i < n; ++i)
      approx += q.getWeights()(i) * std::exp(q.getNodes()(i)) *
                std::cos(M_PI * q.getNodes()(i) / 2.0);

    EXPECT_NEAR(approx, exact, 1e-6) << "n = " << n;
  }
}

// =============================================================================
// TriangleQuadrature fixture
// =============================================================================
class TriangleQuadratureTest : public testing::Test {
 protected:
  TriangleQuadrature TQ;
  static constexpr double tol = 1e-9;
};

// Highest rule index Burkardt's Dunavant table supports.
const int kMaxRule = dunavant_rule_num();

double Factorial(int n) {
  double result = 1.0;
  for (int i = 2; i <= n; ++i) result *= i;
  return result;
}

// Closed-form integral of x^a * y^b over the reference triangle with
// vertices (0,0), (1,0), (0,1).
double tri_exact_solution(int a, int b) {
  return Factorial(a) * Factorial(b) / Factorial(a + b + 2);
}

// Quadrature approximation of the same integral: reference-triangle area
// (0.5) times the barycentric-weighted sum.
double tri_approx_solution(const Eigen::MatrixX2d& pts,
                           const Eigen::VectorXd& w, int a, int b) {
  double sum = 0.0;
  for (int i = 0; i < w.size(); ++i) {
    sum += w(i) * std::pow(pts(i, 0), a) * std::pow(pts(i, 1), b);
  }
  return 0.5 * sum;
}

// =============================================================================
// Test 1 - Range validation for Dunavant rule index
// =============================================================================
TEST_F(TriangleQuadratureTest, RangeValidation) {
  EXPECT_THROW(TQ.compute(0), std::invalid_argument);
  EXPECT_THROW(TQ.compute(-1), std::invalid_argument);
  EXPECT_THROW(TQ.compute(kMaxRule + 1), std::invalid_argument);
}

// =============================================================================
// Test 2 - points.rows() matches known Dunavant suborder counts (rules 1-5)
// =============================================================================
TEST_F(TriangleQuadratureTest, DunavantOrderCounts) {
  const std::vector<std::pair<int, int>> ruleToOrder = {
      {1, 1}, {2, 3}, {3, 4}, {4, 6}, {5, 7}};

  for (const auto& [rule, expectedOrder] : ruleToOrder) {
    TQ.compute(rule);
    EXPECT_EQ(TQ.getPoints().rows(), expectedOrder) << "Rule " << rule;
  }
}

// =============================================================================
// Test 3 - points() and weights() sizes are mutually consistent
// =============================================================================
TEST_F(TriangleQuadratureTest, ConsistentSizes) {
  for (int rule = 1; rule <= kMaxRule; ++rule) {
    TQ.compute(rule);
    EXPECT_EQ(TQ.getPoints().rows(), TQ.getWeights().size()) << "Rule " << rule;
    EXPECT_EQ(TQ.getPoints().cols(), 2) << "Rule " << rule;
  }
}

// =============================================================================
// Test 4 - Quadrature integrates the constant function exactly (area check)
// =============================================================================
TEST_F(TriangleQuadratureTest, IntegratesConstantExactly) {
  for (int rule = 1; rule <= kMaxRule; ++rule) {
    TQ.compute(rule);
    EXPECT_NEAR(tri_approx_solution(TQ.getPoints(), TQ.getWeights(), 0, 0),
                tri_exact_solution(0, 0), tol)
        << "Rule " << rule;
  }
}

// =============================================================================
// Test 5 - Quadrature integrates all monomials up to its exactness degree
// =============================================================================
TEST_F(TriangleQuadratureTest, IntegratesMonomialsExactly) {
  for (int rule : {1, 2, 3, 4, 5, 7, 13}) {
    TQ.compute(rule);

    for (int a = 0; a <= rule; ++a) {
      for (int b = 0; a + b <= rule; ++b) {
        EXPECT_NEAR(tri_approx_solution(TQ.getPoints(), TQ.getWeights(), a, b),
                    tri_exact_solution(a, b), tol)
            << "Rule " << rule << ", monomial x^" << a << " y^" << b;
      }
    }
  }
}

// =============================================================================
// QuadQuadrature fixture
// =============================================================================
class QuadQuadratureTest : public testing::Test {
 protected:
  QuadQuadrature QQ;
  static constexpr double tol = 1e-9;
};

// Closed-form integral of x^a * y^b over the reference square [-1,1] x [-1,1].
double quad_exact_solution(int a, int b) {
  auto dim1_solution = [](int p) {
    return (1.0 - std::pow(-1.0, p + 1)) / (p + 1.0);
  };
  return dim1_solution(a) * dim1_solution(b);
}

// Quadrature approximation of x^a * y^b over the reference square.
double quad_approx_solution(const Eigen::MatrixX2d& pts,
                            const Eigen::VectorXd& w, int a, int b) {
  double sum = 0.0;
  for (int i = 0; i < w.size(); ++i) {
    sum += w(i) * std::pow(pts(i, 0), a) * std::pow(pts(i, 1), b);
  }
  return sum;
}

// =============================================================================
// Test 1 - Range validation for tensor-product rule
// =============================================================================
TEST_F(QuadQuadratureTest, RangeValidation) {
  EXPECT_THROW(QQ.compute(0), std::invalid_argument);
  EXPECT_THROW(QQ.compute(-1), std::invalid_argument);
}

// =============================================================================
// Test 2 - points_.rows() equals n^2 for n points per axis
// =============================================================================
TEST_F(QuadQuadratureTest, NumPointsMatchesN2) {
  for (int n = 1; n <= 5; ++n) {
    QQ.compute(n);
    EXPECT_EQ(QQ.getPoints().rows(), n * n) << "n " << n;
  }
}

// =============================================================================
// Test 3 - points() and weights() sizes are mutually consistent
// =============================================================================
TEST_F(QuadQuadratureTest, PointsAndWeightsSizesConsistent) {
  for (int n = 1; n <= 10; ++n) {
    QQ.compute(n);
    EXPECT_EQ(QQ.getPoints().rows(), QQ.getWeights().size()) << "n " << n;
    EXPECT_EQ(QQ.getPoints().cols(), 2) << "n " << n;
  }
}

// =============================================================================
// Test 4 - Quadrature integrates the constant function exactly (area check)
// =============================================================================
TEST_F(QuadQuadratureTest, IntegratesConstantExactly) {
  for (int n = 1; n <= 10; ++n) {
    QQ.compute(n);
    EXPECT_NEAR(quad_approx_solution(QQ.getPoints(), QQ.getWeights(), 0, 0),
                quad_exact_solution(0, 0), tol)
        << "n " << n;
  }
}

// =============================================================================
// Test 5 - Quadrature integrates separable monomials up to degree 2n-1 exactly
// =============================================================================
TEST_F(QuadQuadratureTest, IntegratesSeparableMonomialsExactly) {
  for (int n = 1; n <= 4; ++n) {
    QQ.compute(n);

    for (int a = 0; a <= 2 * n - 1; ++a) {
      for (int b = 0; b <= 2 * n - 1; ++b) {
        EXPECT_NEAR(quad_approx_solution(QQ.getPoints(), QQ.getWeights(), a, b),
                    quad_exact_solution(a, b), tol)
            << "n " << n << ", monomial x^" << a << " y^" << b;
      }
    }
  }
}