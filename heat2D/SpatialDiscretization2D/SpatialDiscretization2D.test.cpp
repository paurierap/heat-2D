#include <cmath>
#include <Eigen/Sparse>
#include <functional>
#include <gtest/gtest.h>

#include "DirichletBoundaryCondition.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "FiniteDifference2D.hpp"
#include "StructuredMesh2D.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Helper: solves for the approximation and compares it with the exact solution
// =============================================================================
double solve_and_get_error(spatial::SpatialDiscretization2D& sd, const spatial::Mesh2D& mesh, std::function<double (double, double)> solution)
{
    sd.discretize();

    Eigen::VectorXd sol = sd.solveSteadyState();
    Eigen::VectorXd exact(sol.size());

    int j = 0;
    for (const auto& node : mesh.getNodes()) exact[j++] = solution(node.x_, node.y_);

    return (exact - sol).lpNorm<Eigen::Infinity>();
};

// =============================================================================
// Test 1 - Check some coefficients of the Laplacian to ensure correct 
//          implementation
// =============================================================================
TEST(FiniteDifference2D, LaplacianComponents) 
{
    constexpr int nx = 4, ny = 5;
    const spatial::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);

    // Define BCs
    spatial::BoundaryConditions bc;
    auto zeroBC = [](double, double, double){return 0.0;};
    bc[spatial::DomainSide::Left] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Top] = 
    std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);

    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Discretise PDE
    constexpr double alpha = 1.0;
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);

    fd.discretize();
    const Eigen::SparseMatrix<double>& A = fd.getMatrix();
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
// =============================================================================
// Test 2 - Verify consistency of the discrete Laplacian operator for a harmonic 
//          function. For u(x,y) = x^2 - y^2, -Δu = 0.
// =============================================================================
TEST(FiniteDifference2D, LaplacianVanishes) 
{
    constexpr int nx = 21, ny = 21;
    const spatial::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);
    
    // Define BCs
    auto leftBC = [](double x, double y, double t){return - y * y;};
    auto rightBC = [](double x, double y, double t){return 1 - y * y;};
    auto bottomBC = [](double x, double y, double t){return x * x;};
    auto topBC = [](double x, double y, double t){return x * x - 1;};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_shared<spatial::DirichletBoundaryCondition>(leftBC);
    bc[spatial::DomainSide::Right] = std::make_shared<spatial::DirichletBoundaryCondition>(rightBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(bottomBC);
    bc[spatial::DomainSide::Top] = std::make_shared<spatial::DirichletBoundaryCondition>(topBC);
    
    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Exact harmonic solution (quadratic => exact for 2nd-order FD)
    auto solution = [](double x, double y){return x * x - y * y;};

    // Discretize PDE
    constexpr double alpha = 1.0;
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);
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

// =============================================================================
// Test 3 - Verify the expected convergence rate (2nd-order) to solve the
//          Laplace equation with Dirichlet BCs in all sides. 
//    
// For u(x,y) = sin(πx/Lx) * sinh(πy/Lx), -Δu = 0. Imposing Dirichlet BCs at all 
// sides leads to u_left = u_right = u_bottom = 0, u_top = sin(πx/Lx) * 
// sinh(πLy/Lx)
//    
// Two different mesh sizes are used to test convergence, with 
// h_fine = 0.5 * h_coarse.      
// =============================================================================
TEST(FiniteDifference2D, LaplaceDirichletBCconvergence) 
{
    constexpr int n_coarse = 51; 
    constexpr int n_fine = 101;
    constexpr double Lx = 1, Ly = 1;

    const spatial::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, n_coarse, n_coarse);
    const spatial::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, n_fine, n_fine);

    // Define BCs
    auto zeroBC = [](double x, double y, double t){return 0;};
    auto topBC = [&](double x, double y, double t){return std::sin(M_PI * x / Lx) * std::sinh(M_PI * Ly / Lx);};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Top] = std::make_shared<spatial::DirichletBoundaryCondition>(topBC);
    
    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Exact solution
    auto solution = [&](double x, double y){return std::sin(M_PI * x / Lx) * std::sinh(M_PI * y / Lx);};

    // Discretise PDE
    constexpr double alpha = 1.0;
    spatial::FiniteDifference2D fd_coarse(alpha, mesh_coarse, bc, source);
    spatial::FiniteDifference2D fd_fine(alpha, mesh_fine, bc, source);

    double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
    double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
    double h_coarse = mesh_coarse.getDx();
    double h_fine = mesh_fine.getDx();

    // Verify expected convergence rate
    double convergence_rate = std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
    EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test 4 - Verify the expected convergence rate (2nd-order) to solve the 
//          Laplace equation with mixed BCs.  
//    
// For u(x,y) = sin(πx/Lx) * cosh(πy/Lx), -Δu = 0. Imposing Dirichlet BCs on the 
// right, left, and bottom sides leads to u_left = u_right = 0, and u_bottom = 
// = (πx/Lx). Imposing a Neumann BC on the top reads 
// du/dy|_top = π/Lx * sin(πx/Lx) * sinh(πy/Lx)
//    
// Two different mesh sizes are used to test convergence, with 
// h_fine = 0.5 * h_coarse.      
// =============================================================================
TEST(FiniteDifference2D, LaplaceMixedBCconvergence)
{
    constexpr int n_coarse = 51;
    constexpr int n_fine = 101;
    constexpr double Lx = 2.0, Ly = 3.0;

    const spatial::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, n_coarse, n_coarse);
    const spatial::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, n_fine, n_fine);

    // Define BCs
    auto zeroBC = [](double, double, double){return 0.0;};
    auto bottomBC = [&](double x, double, double){return std::sin(M_PI * x /Lx);};
    auto topBC = [&](double x, double, double){return M_PI / Lx * std::sin(M_PI * x / Lx) * std::sinh(M_PI * Ly / Lx);};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(bottomBC);
    bc[spatial::DomainSide::Top] = std::make_shared<spatial::NeumannBoundaryCondition>(topBC);

    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Exact solution
    auto solution = [&](double x, double y){return std::sin(M_PI * x / Lx) * std::cosh(M_PI * y / Lx);};

    // Discretise PDE
    constexpr double alpha = 1.0;
    spatial::FiniteDifference2D fd_coarse(alpha, mesh_coarse, bc, source);
    spatial::FiniteDifference2D fd_fine(alpha, mesh_fine, bc, source);

    double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
    double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
    double h_coarse = mesh_coarse.getDx();
    double h_fine = mesh_fine.getDx();

    // Verify expected convergence rate
    double convergence_rate = std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
    EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test 5 - Verify that the Laplacian matrix with pure Neumann BCs has a 
//          constant vector in its nullspace. That is, A * ones = 0.
// =============================================================================
TEST(FiniteDifference2D, LaplaceNullSpace)
{
    constexpr int n = 101;                  
    constexpr double Lx = 2.0, Ly = 3.0;    
    spatial::StructuredMesh2D mesh(0, Lx, 0, Ly, n, n);

    // Define BCs
    auto zeroBC = [](double, double, double){return 0.0;};
    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_shared<spatial::NeumannBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right] = std::make_shared<spatial::NeumannBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::NeumannBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Top] = std::make_shared<spatial::NeumannBoundaryCondition>(zeroBC);

    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Discretise PDE
    constexpr double alpha = 1.0;
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);
    fd.discretize();

    const Eigen::SparseMatrix<double>& A = fd.getMatrix();
    Eigen::VectorXd ones = Eigen::VectorXd::Constant(A.cols(), 1.0);

    EXPECT_NEAR((A * ones).lpNorm<Eigen::Infinity>(), 0.0, 1e-12);
}

// =============================================================================
// Test 6 - Verify the expected convergence rate (2nd-order) to solve the 
//          Poisson equation with mixed BCs.  
//    
// For u(x,y) = exp(-x²) * sin(πy), -Δu = -(4x² - 2 - π²) * exp(-x²) * sin(πy). 
// Imposing Dirichlet BCs on the bottom and top sides leads to u_bottom = u_top 
// = 0. Imposing Neumann BCs on the left and right reads du/dy|_left = 0, and
// du/dy|_right = -2 / exp(1) * sin(πy).
//
// Two different mesh sizes are used to test convergence, with 
// h_fine = 0.5 * h_coarse.      
// =============================================================================
TEST(FiniteDifference2D, PoissonMixedBCconvergence)
{
    constexpr int n_coarse = 51;
    constexpr int n_fine = 101;

    const spatial::StructuredMesh2D mesh_coarse(0, 1, 0, 1, n_coarse, n_coarse);
    const spatial::StructuredMesh2D mesh_fine(0, 1, 0, 1, n_fine, n_fine);

    // Define BCs
    auto zeroBC = [](double, double, double){return 0.0;};
    auto rightBC = [](double, double y, double){return -2 * std::exp(-1) * std::sin(M_PI * y);};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_shared<spatial::NeumannBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right] = std::make_shared<spatial::NeumannBoundaryCondition>(rightBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Top] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);

    // Exact solution
    auto solution = [](double x, double y){return std::exp(- x * x) * std::sin(M_PI * y);};

    // Source term:
    auto source = [&](double x, double y, double){return -(4 * x * x - 2 - M_PI * M_PI) * solution(x,y);};

    // Discretize PDE
    constexpr double alpha = 1.0;
    spatial::FiniteDifference2D fd_coarse(alpha, mesh_coarse, bc, source);
    spatial::FiniteDifference2D fd_fine(alpha, mesh_fine, bc, source);

    double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
    double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
    double h_coarse = mesh_coarse.getDx();
    double h_fine = mesh_fine.getDx();

    double convergence_rate = std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
    EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test 7 - Verify that the Poisson equation is solved with mixed BCs and an 
//          anisotropic grid.
//
// For u(x,y,t) = log(sin²(x * y) + 1), -Δu = -(x² + y²) * (3 * cos(2 * x * y) 
// - 1) / (1 + sin²(x * y))².  Imposing Dirichlet BCs at the left and right 
// sides leads to u_left = 0 and u_right = log(sin²(2 * y) + 1). Imposing 
// Neumann BCs on the bottom and top sides leads to du/dy|_bottom = 0, and 
// du/dy|_top = x * sin(2 * x) / (sin²(x) + 1).
//
// With the choice of parameters, O(error) ≈ O(dx²) ≈ 4e-4 < 1e-3.
// =============================================================================
TEST(FiniteDifference2D, PoissonMixedBCAnisotropicGrid)
{
    constexpr int nx = 101, ny = 51;
    const spatial::StructuredMesh2D mesh(0, 2, 0, 1, nx, ny);

    // Define BCs
    spatial::BoundaryConditions bc;
    auto zeroBC = [](double, double, double){return 0.0;};
    auto rightBC = [](double, double y, double)
    {
        double s = std::sin(2 * y);
        return std::log(s * s + 1);
    };
    auto topBC = [](double x, double, double)
    {
        double s = std::sin(x);
        return x * std::sin(2 * x) / (s * s + 1);
    };

    bc[spatial::DomainSide::Left] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right] = std::make_shared<spatial::DirichletBoundaryCondition>(rightBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::NeumannBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Top] = std::make_shared<spatial::NeumannBoundaryCondition>(topBC);

    // Exact solution
    auto exact = [](double x, double y)
    {
        double s = std::sin(x * y);
        return std::log(s * s + 1);
    };

    // Source term
    auto source = [](double x, double y, double)
    {
        double s = std::sin(x * y);
        double d = 1.0 + s * s;
        return -(x * x + y * y) * (3.0 * std::cos(2 * x * y) - 1.0) / (d * d);
    };

    // Discretize PDE
    constexpr double alpha = 1.0;
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);

    double err = solve_and_get_error(fd, mesh, exact);

    EXPECT_LT(err, 1e-3);
}