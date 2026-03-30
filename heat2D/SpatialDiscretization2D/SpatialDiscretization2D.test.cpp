#include <cmath>
#include <Eigen/Eigenvalues>
#include <functional>
#include <gtest/gtest.h>

#include "DirichletBoundaryCondition.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "FiniteDifference2D.hpp"
#include "StructuredMesh2D.hpp"

double solve_and_get_error(spatial::SpatialDiscretization2D& sd, spatial::Mesh2D& mesh, std::function<double (double, double)> solution)
{
    sd.discretize();

    Eigen::VectorXd sol = sd.solve();
    Eigen::VectorXd exact(sol.size());

    int j = 0;
    for (const auto& node : mesh.getNodes()) exact[j++] = solution(node.x_, node.y_);

    Eigen::VectorXd residual = exact - sol;
    return residual.lpNorm<Eigen::Infinity>();
};

TEST(FiniteDifference2D, LaplacianComponents) 
{
    // This test checks the individual coefficients of the Laplacian matrix to ensure consistency with the 2nd order 5-element stencil.

    int nx = 4, ny = 5;
    spatial::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);

    // Boundary Conditions
    spatial::BoundaryConditions bc;
    std::function<double (double, double, double)> f = [](double x, double y, double t)
    {return 0;};
    bc[spatial::DomainSide::Left] = std::make_unique<spatial::DirichletBoundaryCondition>(f);
    bc[spatial::DomainSide::Right] = std::make_unique<spatial::DirichletBoundaryCondition>(f);
    bc[spatial::DomainSide::Top] = 
    std::make_unique<spatial::DirichletBoundaryCondition>(f);
    bc[spatial::DomainSide::Bottom] = std::make_unique<spatial::DirichletBoundaryCondition>(f);

    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Discretise PDE
    spatial::FiniteDifference2D fd(1.0, mesh, bc, source);

    fd.discretize();
    const auto& A = fd.getMatrix();
    double dx = mesh.getDx();
    double dy = mesh.getDy();

    ASSERT_GT(A.rows(), 5);
    EXPECT_DOUBLE_EQ(A.coeff(0, 0), -2.0/(dx*dx) - 2.0/(dy*dy));
    EXPECT_DOUBLE_EQ(A.coeff(0, 1), 1.0/(dx*dx));
    EXPECT_DOUBLE_EQ(A.coeff(0, 2), 1.0/(dy*dy));
    EXPECT_DOUBLE_EQ(A.coeff(5, 5), -2.0/(dx*dx) - 2.0/(dy*dy));
    EXPECT_DOUBLE_EQ(A.coeff(5, 4), 1.0/(dx*dx));
    EXPECT_DOUBLE_EQ(A.coeff(5, 3), 1.0/(dy*dy));
}

TEST(FiniteDifference2D, LaplacianVanishes) 
{
    // This test verifies consistency of the discrete Laplacian operator for a harmonic function. For u(x,y) = x^2 - y^2, we have -Δu = 0.
    int nx = 21, ny = 21;
    spatial::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);
    
    // Define BCs
    std::function<double (double, double, double)> leftBC = [](double x, double y, double t){return - y * y;};
    std::function<double (double, double, double)> rightBC = [](double x, double y, double t){return 1 - y * y;};
    std::function<double (double, double, double)> bottomBC = [](double x, double y, double t){return x * x;};
    std::function<double (double, double, double)> topBC = [](double x, double y, double t){return x * x - 1;};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_unique<spatial::DirichletBoundaryCondition>(leftBC);
    bc[spatial::DomainSide::Right] = std::make_unique<spatial::DirichletBoundaryCondition>(rightBC);
    bc[spatial::DomainSide::Bottom] = std::make_unique<spatial::DirichletBoundaryCondition>(bottomBC);
    bc[spatial::DomainSide::Top] = std::make_unique<spatial::DirichletBoundaryCondition>(topBC);
    
    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Exact harmonic solution (quadratic => exact for 2nd-order FD)
    std::function<double (double, double)> solution = [](double x, double y){return x * x - y * y;};

    // Discretize PDE
    spatial::FiniteDifference2D fd(1.0, mesh, bc, source);
    fd.discretize();
    fd.updateRHS();

    // A: interior Laplacian matrix
    // b: boundary contribution from Dirichlet nodes
    const auto& A = fd.getMatrix();
    Eigen::VectorXd b = fd.getVector(), exact(A.cols());

    ASSERT_EQ(A.cols(), mesh.getInnerNodes().size());
    ASSERT_EQ(b.size(), A.rows());

    int j = 0;
    auto nodes = mesh.getNodes();
    for (int nodeID : mesh.getInnerNodes()) exact[j++] = solution(nodes[nodeID].x_, nodes[nodeID].y_);

    // Discrete residual should vanish up to roundoff
    Eigen::VectorXd res = A * exact + b;
    EXPECT_LT(res.lpNorm<Eigen::Infinity>(), 1e-12);
}

TEST(FiniteDifference2D, Laplace_DirichletBC_SinSinh_Convergence) 
{
    /* 
    This test verifies the expected convergence rate (2nd-order) to solve the Laplace equation with Dirichlet BCs in all sides. 
    
    For u(x,y) = sin(πx/Lx) * sinh(πy/Lx), we have -Δu = 0.

    We can impose Dirichlet BCs at all sides where u_left = u_right = u_bottom = 0, but u_top = sin(πx/Lx)sinh(πLy/Lx).
    
    We use two different mesh sizes to test convergence, with h_fine = 0.5 * h_coarse.
    */
    int n_coarse = 51; 
    int n_fine = 101;
    double Lx = 1, Ly = 1;

    spatial::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, n_coarse, n_coarse);
    spatial::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, n_fine, n_fine);

    // Define BCs
    std::function<double (double, double, double)> leftBC = [](double x, double y, double t){return 0;};
    std::function<double (double, double, double)> rightBC = [](double x, double y, double t){return 0;};
    std::function<double (double, double, double)> bottomBC = [](double x, double y, double t){return 0;};
    std::function<double (double, double, double)> topBC = [&](double x, double y, double t){return std::sin(M_PI * x / Lx) * std::sinh(M_PI * Ly / Lx);};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_unique<spatial::DirichletBoundaryCondition>(leftBC);
    bc[spatial::DomainSide::Right] = std::make_unique<spatial::DirichletBoundaryCondition>(rightBC);
    bc[spatial::DomainSide::Bottom] = std::make_unique<spatial::DirichletBoundaryCondition>(bottomBC);
    bc[spatial::DomainSide::Top] = std::make_unique<spatial::DirichletBoundaryCondition>(topBC);
    
    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Exact solution
    std::function<double (double, double)> solution = [&](double x, double y){return std::sin(M_PI * x / Lx) * std::sinh(M_PI * y / Lx);};

    // Discretise PDE
    spatial::FiniteDifference2D fd_coarse(1.0, mesh_coarse, bc, source);
    spatial::FiniteDifference2D fd_fine(1.0, mesh_fine, bc, source);

    double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
    double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
    double h_coarse = mesh_coarse.getDx();
    double h_fine = mesh_fine.getDx();

    // Verify expected convergence rate
    double convergence_rate = std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
    EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

TEST(FiniteDifference2D, Laplace_MixedBC_SinCosh_Convergence)
{
    /* 
    This test verifies the expected convergence rate (2nd-order) to solve the Laplace equation with mixed BCs. 
    
    For u(x,y) = sin(πx/Lx) * cosh(πy/Lx), we have -Δu = 0.

    We can impose Dirichlet BCs for u_left = u_right = 0, but u_bottom = sin(πx/Lx). In addition, we have du/dy|_top = π/Lx * sin(πx/Lx) * sinh(πy/Lx)
    
    We use two different mesh sizes to test convergence, with h_fine = 0.5 * h_coarse.
    */
    int n_coarse = 51;
    int n_fine = 101;
    double Lx = 2.0, Ly = 3.0;

    spatial::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, n_coarse, n_coarse);
    spatial::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, n_fine, n_fine);

    // Define BCs
    auto leftBC = [](double, double, double){return 0.0;};
    auto rightBC = [](double, double, double){return 0.0;};
    auto bottomBC = [&](double x, double, double){return std::sin(M_PI * x /Lx);};
    auto topBC = [&](double x, double, double){return M_PI / Lx * std::sin(M_PI * x / Lx) * std::sinh(M_PI * Ly / Lx);};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_unique<spatial::DirichletBoundaryCondition>(leftBC);
    bc[spatial::DomainSide::Right] = std::make_unique<spatial::DirichletBoundaryCondition>(rightBC);
    bc[spatial::DomainSide::Bottom] = std::make_unique<spatial::DirichletBoundaryCondition>(bottomBC);
    bc[spatial::DomainSide::Top] = std::make_unique<spatial::NeumannBoundaryCondition>(topBC);

    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Exact solution
    std::function<double (double, double)> solution = [&](double x, double y){return std::sin(M_PI * x / Lx) * std::cosh(M_PI * y / Lx);};

    // Discretise PDE
    spatial::FiniteDifference2D fd_coarse(1.0, mesh_coarse, bc, source);
    spatial::FiniteDifference2D fd_fine(1.0, mesh_fine, bc, source);

    double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
    double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
    double h_coarse = mesh_coarse.getDx();
    double h_fine = mesh_fine.getDx();

    // Verify expected convergence rate
    double convergence_rate = std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
    EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

TEST(FiniteDifference2D, Laplace_NullSpace)
{
    /* 
    This test verifies that the Laplacian matrix with pure Neumann BCs has a constant vector in its nullspace. That is, A * ones = 0.
    */
    int n = 101;                  
    double Lx = 2.0, Ly = 3.0;    
    spatial::StructuredMesh2D mesh(0, Lx, 0, Ly, n, n);

    // Define BCs
    auto zeroNeumann = [](double, double, double){return 0.0;};
    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_unique<spatial::NeumannBoundaryCondition>(zeroNeumann);
    bc[spatial::DomainSide::Right] = std::make_unique<spatial::NeumannBoundaryCondition>(zeroNeumann);
    bc[spatial::DomainSide::Bottom] = std::make_unique<spatial::NeumannBoundaryCondition>(zeroNeumann);
    bc[spatial::DomainSide::Top] = std::make_unique<spatial::NeumannBoundaryCondition>(zeroNeumann);

    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Discretise PDE
    spatial::FiniteDifference2D fd(1.0, mesh, bc, source);
    fd.discretize();

    const auto& A = fd.getMatrix();
    Eigen::VectorXd ones = Eigen::VectorXd::Constant(A.cols(), 1.0);

    EXPECT_NEAR((A * ones).lpNorm<Eigen::Infinity>(), 0.0, 1e-12);
}


TEST(FiniteDifference2D, Poisson_MixedBC_SquareExpSin_Convergence)
{
    /* This test verifies the expected convergence rate (2nd-order) to solve the Poisson equation with mixed BCs.
    
    For u(x,y) = exp(-x²) * sin(πy), we have -Δu = -(4x² - 2 - π²) * exp(-x²) * sin(πy).

    We can impose Dirichlet BCs for u_top = u_bottom = 0. In addition, we have du/dy|_left = 0, and du/dy|_right = -2 / exp(1) * sin(πy).
    
    We use two different mesh sizes to test convergence, with h_fine = 0.5 * h_coarse.
    */
    int n_coarse = 51;
    int n_fine = 101;

    spatial::StructuredMesh2D mesh_coarse(0, 1, 0, 1, n_coarse, n_coarse);
    spatial::StructuredMesh2D mesh_fine(0, 1, 0, 1, n_fine, n_fine);

    // Define BCs
    auto leftBC = [](double, double, double){return 0.0;};
    auto rightBC = [](double, double y, double){return -2 * std::exp(-1) * std::sin(M_PI * y);};
    auto bottomBC = [](double, double, double){return 0.0;};
    auto topBC = [](double, double, double){return 0.0;};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_unique<spatial::NeumannBoundaryCondition>(leftBC);
    bc[spatial::DomainSide::Right] = std::make_unique<spatial::NeumannBoundaryCondition>(rightBC);
    bc[spatial::DomainSide::Bottom] = std::make_unique<spatial::DirichletBoundaryCondition>(bottomBC);
    bc[spatial::DomainSide::Top] = std::make_unique<spatial::DirichletBoundaryCondition>(topBC);

    // Exact solution
    std::function<double (double, double)> solution = [](double x, double y){return std::exp(- x * x) * std::sin(M_PI * y);};

    // Source term:
    auto source = [&](double x, double y, double){return -(4 * x * x - 2 - M_PI * M_PI) * solution(x,y);};

    // Discretise PDE
    spatial::FiniteDifference2D fd_coarse(1.0, mesh_coarse, bc, source);
    spatial::FiniteDifference2D fd_fine(1.0, mesh_fine, bc, source);

    double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
    double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
    double h_coarse = mesh_coarse.getDx();
    double h_fine = mesh_fine.getDx();

    double convergence_rate = std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
    EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}