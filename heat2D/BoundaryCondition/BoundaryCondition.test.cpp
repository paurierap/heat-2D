#include <gtest/gtest.h>
#include <Eigen/Sparse>
#include <functional>
#include <vector>

#include "DirichletBoundaryCondition.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "Mesh2D.hpp"

class BoundaryConditionTest : public testing::Test
{
    protected: 
        std::function<double(double, double, double)> f;
        spatial::DirichletBoundaryCondition DirichletBC;
        spatial::NeumannBoundaryCondition NeumannBC;

    BoundaryConditionTest() 
    : f([](double x, double y, double t){ return x - y; }), DirichletBC(spatial::Side::Left, [](double x, double y, double t){ return x + y; }), NeumannBC(spatial::Side::Right, f) {};
};

TEST_F(BoundaryConditionTest, Constructor) 
{
    EXPECT_DOUBLE_EQ(DirichletBC.f(1,2,30), 3);
    EXPECT_EQ(DirichletBC.getSide(), spatial::Side::Left);

    EXPECT_DOUBLE_EQ(NeumannBC.f(1,2,1), -1);
    EXPECT_EQ(NeumannBC.getSide(), spatial::Side::Right);
}

TEST_F(BoundaryConditionTest, applyBCs) 
{
    spatial::StructuredMesh2D mesh{0,1,0,1,11,11};
    std::vector<Eigen::Triplet<double>> triplets;

    DirichletBC.applyBoundaryCondition(mesh, triplets);
    NeumannBC.applyBoundaryCondition(mesh, triplets);
}