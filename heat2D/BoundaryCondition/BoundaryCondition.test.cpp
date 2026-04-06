#include <gtest/gtest.h>
#include <functional>
#include <math.h>

#include "DirichletBoundaryCondition.hpp"
#include "NeumannBoundaryCondition.hpp"

class BoundaryConditionTest : public testing::Test
{
    protected: 
        std::function<double(double, double, double)> f;
        spatial::DirichletBoundaryCondition DirichletBC;
        spatial::NeumannBoundaryCondition NeumannBC;

    BoundaryConditionTest() 
    : f([](double x, double y, double t){ return x - y; }), 
    DirichletBC([](double x, double y, double t){ return x + y; }), 
    NeumannBC(f) 
    {};
};

// =============================================================================
// Test 1 — Constructor
// =============================================================================
TEST_F(BoundaryConditionTest, Constructor) 
{
    EXPECT_DOUBLE_EQ(DirichletBC.f(1,2,30), 3);
    EXPECT_EQ(DirichletBC.getType(), spatial::BoundaryConditionType::Dirichlet);

    EXPECT_DOUBLE_EQ(NeumannBC.f(1,2,1), -1);
    EXPECT_EQ(NeumannBC.getType(), spatial::BoundaryConditionType::Neumann);
}