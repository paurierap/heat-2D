#include <gtest/gtest.h>
#include <math.h>

#include <functional>

#include "DirichletBoundaryCondition.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "RobinBoundaryCondition.hpp"

using namespace heat2d::bc;

class BoundaryConditionTest : public testing::Test {
 protected:
  std::function<double(double, double, double)> f;
  std::function<double(double, double, double)> u_coeff{
      [](double x, double y, double t) { return 1.0; }};
  std::function<double(double, double, double)> du_coeff{
      [](double x, double y, double t) { return 2.0; }};
  DirichletBoundaryCondition DirichletBC;
  NeumannBoundaryCondition NeumannBC;
  RobinBoundaryCondition RobinBC;

  BoundaryConditionTest()
      : f([](double x, double y, double t) { return x - y; }),
        DirichletBC([](double x, double y, double t) { return x + y; }),
        NeumannBC(f),
        RobinBC(u_coeff, du_coeff, [](double x, double y, double t) {
          return x * std::exp(t - y);
        }) {};
};

// =============================================================================
// Test 1 — Constructor
// =============================================================================
TEST_F(BoundaryConditionTest, Constructor) {
  EXPECT_DOUBLE_EQ(DirichletBC.f(1, 2, 30), 3);
  EXPECT_EQ(DirichletBC.getType(), BoundaryConditionType::Dirichlet);

  EXPECT_DOUBLE_EQ(NeumannBC.f(1, 2, 1), -1);
  EXPECT_EQ(NeumannBC.getType(), BoundaryConditionType::Neumann);

  EXPECT_DOUBLE_EQ(RobinBC.f(1, 2, 1), 1 * std::exp(1 - 2));
  EXPECT_EQ(RobinBC.getType(), BoundaryConditionType::Robin);
}