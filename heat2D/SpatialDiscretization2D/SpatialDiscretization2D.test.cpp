#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>

#include "FiniteDifference2D.hpp"
#include "StructuredMesh2D.hpp"

TEST(FiniteDifference2D, Laplacian3x3) 
{
    int nx = 3, ny = 3;
    spatial::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);
    spatial::FiniteDifference2D fd(mesh);

    fd.applyLaplacian();
    const auto& A = fd.getMatrix();
    double dx = mesh.getDx(), dy = mesh.getDy();

    EXPECT_DOUBLE_EQ(A.coeff(4, 4 - 1), 1.0/(dx*dx));
    EXPECT_DOUBLE_EQ(A.coeff(4, 4 + 1), 1.0/(dx*dx));
    EXPECT_DOUBLE_EQ(A.coeff(4, 4), -2.0/(dx*dx) - 2.0/(dy*dy));
    EXPECT_DOUBLE_EQ(A.coeff(4, 4 - nx), 1.0/(dy*dy));
    EXPECT_DOUBLE_EQ(A.coeff(4, 4 + nx), 1.0/(dy*dy));
}

TEST(FiniteDifference2D, LaplacianOfQuadratic) 
{
    int nx = 11, ny = 11;
    spatial::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);
    spatial::FiniteDifference2D fd(mesh);

    fd.applyLaplacian();
    const auto& A = fd.getMatrix();

    // Build vector u(x,y) = x^2 + y^2
    Eigen::VectorXd u(mesh.getNodes().size());
    for (int i = 0; i < mesh.getNodes().size(); ++i) 
    {
        const auto& node = mesh.getNodes()[i];
        u[i] = node.x_ * node.x_ + node.y_ * node.y_;
    }

    // Compute Laplacian * u
    Eigen::VectorXd Lu = A * u;

    for (int id : mesh.getInnerNodes()) EXPECT_NEAR(Lu[id], 4.0, 1e-10);
}

TEST(FiniteDifference2D, MatrixIsSymmetric) 
{
    spatial::StructuredMesh2D mesh(0,1,0,1,3,3);
    spatial::FiniteDifference2D fd(mesh);

    fd.applyLaplacian();
    const auto& A = fd.getMatrix();

    Eigen::MatrixXd dense = Eigen::MatrixXd(A);
    EXPECT_NEAR((dense - dense.transpose()).norm(), 0.0, 1e-12);
}

TEST(FiniteDifference2D, EigenvaluesNonPositive) 
{
    spatial::StructuredMesh2D mesh(0,1,0,1,5,5);

    spatial::FiniteDifference2D fd(mesh);
    fd.applyLaplacian();
    const auto& A = fd.getMatrix();

    Eigen::MatrixXd dense = Eigen::MatrixXd(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);

    for (int i = 0; i < es.eigenvalues().size(); ++i) EXPECT_LE(es.eigenvalues()[i], 1e-12); 
}