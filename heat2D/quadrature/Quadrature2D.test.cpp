#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "QuadQuadrature.hpp"
#include "TriangleQuadrature.hpp"

using namespace heat2d::quadrature;

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
double tri_approx_solution(const TriangleQuadrature& q, int a, int b) {
  double sum = 0.0;
  const auto& pts = q.getPoints();
  const auto& w = q.getWeights();
  for (int i = 0; i < q.numPoints(); ++i) {
    sum += w(i) * std::pow(pts(i, 0), a) * std::pow(pts(i, 1), b);
  }
  return 0.5 * sum;
}

// =============================================================================
// TriangleQuadrature fixture
// =============================================================================
class TriangleQuadratureTest : public testing::Test {
 protected:
  static constexpr double tol = 1e-9;
};

// =============================================================================
// Test 1 - Range validation for Dunavant rule index
// =============================================================================
TEST_F(TriangleQuadratureTest, RangeValidation) {
  EXPECT_THROW(TriangleQuadrature(0), std::invalid_argument);
  EXPECT_THROW(TriangleQuadrature(-1), std::invalid_argument);
  EXPECT_THROW(TriangleQuadrature(kMaxRule + 1), std::invalid_argument);
}

// =============================================================================
// Test 2 - numPoints() matches known Dunavant suborder counts (rules 1-5)
// =============================================================================
TEST_F(TriangleQuadratureTest, DunavantOrderCounts) {
  const std::vector<std::pair<int, int>> ruleToOrder = {
      {1, 1}, {2, 3}, {3, 4}, {4, 6}, {5, 7}};

  for (const auto& [rule, expectedOrder] : ruleToOrder) {
    TriangleQuadrature q(rule);
    EXPECT_EQ(q.numPoints(), expectedOrder) << "Rule " << rule;
  }
}

// =============================================================================
// Test 3 - points() and weights() sizes are mutually consistent
// =============================================================================
TEST_F(TriangleQuadratureTest, ConsistentSizes) {
  for (int rule = 1; rule <= kMaxRule; ++rule) {
    TriangleQuadrature q(rule);
    EXPECT_EQ(q.getPoints().rows(), q.numPoints()) << "Rule " << rule;
    EXPECT_EQ(q.getPoints().cols(), 2) << "Rule " << rule;
    EXPECT_EQ(q.getWeights().size(), q.numPoints()) << "Rule " << rule;
  }
}

// =============================================================================
// Test 4 - Quadrature integrates the constant function exactly (area check)
// =============================================================================
TEST_F(TriangleQuadratureTest, IntegratesConstantExactly) {
  for (int rule = 1; rule <= kMaxRule; ++rule) {
    TriangleQuadrature q(rule);
    EXPECT_NEAR(tri_approx_solution(q, 0, 0),
                    tri_exact_solution(0, 0), tol)
        << "Rule " << rule;
  }
}

// =============================================================================
// Test 5 - Quadrature integrates all monomials up to its exactness degree
// =============================================================================
TEST_F(TriangleQuadratureTest, IntegratesMonomialsExactly) {
  for (int rule : {1, 2, 3, 4, 5, 7, 13}) {
    TriangleQuadrature q(rule);

    for (int a = 0; a <= rule; ++a) {
      for (int b = 0; a + b <= rule; ++b) {
        EXPECT_NEAR(tri_approx_solution(q, a, b),
                    tri_exact_solution(a, b), tol)
            << "Rule " << rule << ", monomial x^" << a << " y^" << b;
      }
    }
  }
}

// =============================================================================
// QuadQuadrature helpers
// =============================================================================

// Closed-form integral of x^a * y^b over the reference square [-1,1] x [-1,1].
double quad_exact_solution(int a, int b) {
  auto dim1_solution = [](int p) { return (1.0 - std::pow(-1.0, p + 1)) / (p + 1.0); };
  return dim1_solution(a) * dim1_solution(b);
}

// Quadrature approximation of x^a * y^b over the reference square.
double quad_approx_solution(const QuadQuadrature& q, int a, int b) {
  double sum = 0.0;
  const auto& pts = q.getPoints();
  const auto& w = q.getWeights();
  for (int i = 0; i < q.numPoints(); ++i) {
    sum += w(i) * std::pow(pts(i, 0), a) * std::pow(pts(i, 1), b);
  }
  return sum;
}

// =============================================================================
// QuadQuadrature fixture
// =============================================================================
class QuadQuadratureTest : public testing::Test {
 protected:
  static constexpr double tol = 1e-9;
};

// =============================================================================
// Test 1 - Range validation for tensor-product rule
// =============================================================================
TEST_F(QuadQuadratureTest, RangeValidation) {
  EXPECT_THROW(QuadQuadrature(0), std::invalid_argument);
  EXPECT_THROW(QuadQuadrature(-1), std::invalid_argument);
}

// =============================================================================
// Test 2 - numPoints() equals n^2 for n points per axis
// =============================================================================
TEST_F(QuadQuadratureTest, NumPointsMatchesN2) {
  for (int n = 1; n <= 5; ++n) {
    QuadQuadrature q(n);
    EXPECT_EQ(q.numPoints(), n * n) << "n " << n;
  }
}

// =============================================================================
// Test 3 - points() and weights() sizes are mutually consistent
// =============================================================================
TEST_F(QuadQuadratureTest, PointsAndWeightsSizesConsistent) {
  for (int n = 1; n <= 10; ++n) {
    QuadQuadrature q(n);
    EXPECT_EQ(q.getPoints().rows(), q.numPoints()) << "n " << n;
    EXPECT_EQ(q.getPoints().cols(), 2) << "n " << n;
    EXPECT_EQ(q.getWeights().size(), q.numPoints()) << "n " << n;
  }
}

// =============================================================================
// Test 4 - Quadrature integrates the constant function exactly (area check)
// =============================================================================
TEST_F(QuadQuadratureTest, IntegratesConstantExactly) {
  for (int n = 1; n <= 10; ++n) {
    QuadQuadrature q(n);
    EXPECT_NEAR(quad_approx_solution(q, 0, 0),
                quad_exact_solution(0, 0), tol)
        << "n " << n;
  }
}

// =============================================================================
// Test 5 - Quadrature integrates separable monomials up to degree 2n-1 exactly
// =============================================================================
TEST_F(QuadQuadratureTest, IntegratesSeparableMonomialsExactly) {
  for (int n = 1; n <= 4; ++n) {
    QuadQuadrature q(n);

    for (int a = 0; a <= 2 * n - 1; ++a) {
      for (int b = 0; b <= 2 * n - 1; ++b) {
        EXPECT_NEAR(quad_approx_solution(q, a, b),
                    quad_exact_solution(a, b), tol)
            << "n " << n << ", monomial x^" << a << " y^" << b;
      }
    }
  }
}