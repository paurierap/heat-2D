#include <gtest/gtest.h>

#include <Eigen/Sparse>
#include <cmath>
#include <functional>

#include "BoundaryConditions.hpp"
#include "Solver.hpp"
#include "fem/IsoparametricMapping.hpp"
#include "Mesh.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace heat2d;

// =============================================================================
// Helper: solves for the approximation and compares it with the exact solution
// =============================================================================
double solve_and_get_error(solver::SpatialDiscretization2D& sd,
                           const mesh::Mesh2D& mesh,
                           std::function<double(double, double)> solution) {
  sd.discretize();

  Eigen::VectorXd sol = sd.solveSteadyState();
  Eigen::VectorXd exact(sol.size());

  std::size_t j = 0;
  for (const auto& node : mesh.getNodes())
    exact[j++] = solution(node.x_, node.y_);

  return (exact - sol).lpNorm<Eigen::Infinity>();
};

// =============================================================================
// Test 1 - Check some coefficients of the Laplacian to ensure correct
//          implementation
// =============================================================================
TEST(FiniteDifference2D, LaplacianComponents) {
  constexpr std::size_t nx = 4, ny = 5;
  const mesh::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);

  // Define BCs
  solver::BoundaryConditions bc;
  auto zeroBC = [](double, double, double) { return 0.0; };
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);

  // Source term
  auto source = [](double, double, double) { return 0.0; };

  // Discretize PDE
  auto alpha = [](double, double) { return 1.0; };
  solver::FiniteDifference2D fd(alpha, mesh, bc, source);

  fd.discretize();
  const auto& A = fd.getMatrixK();
  double dx = mesh.getDx();
  double dy = mesh.getDy();

  ASSERT_GT(A.rows(), 5);
  EXPECT_DOUBLE_EQ(A.coeff(0, 0), -2.0 / (dx * dx) - 2.0 / (dy * dy));
  EXPECT_DOUBLE_EQ(A.coeff(0, 1), 1.0 / (dx * dx));
  EXPECT_DOUBLE_EQ(A.coeff(0, 2), 1.0 / (dy * dy));
  EXPECT_DOUBLE_EQ(A.coeff(5, 5), -2.0 / (dx * dx) - 2.0 / (dy * dy));
  EXPECT_DOUBLE_EQ(A.coeff(5, 4), 1.0 / (dx * dx));
  EXPECT_DOUBLE_EQ(A.coeff(5, 3), 1.0 / (dy * dy));
}
// =============================================================================
// Test 2 - Verify consistency of the discrete Laplacian operator for a harmonic
//          function. For u(x,y) = x^2 - y^2, -div(α∇u) = 0.
// =============================================================================
TEST(FiniteDifference2D, LaplacianVanishes) {
  constexpr std::size_t nx = 21, ny = 21;
  const mesh::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);

  // Define BCs
  auto leftBC = [](double x, double y, double t) { return -y * y; };
  auto rightBC = [](double x, double y, double t) { return 1 - y * y; };
  auto bottomBC = [](double x, double y, double t) { return x * x; };
  auto topBC = [](double x, double y, double t) { return x * x - 1; };

  solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(leftBC);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(rightBC);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(bottomBC);
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(topBC);

  // Source term
  auto source = [](double, double, double) { return 0.0; };

  // Exact harmonic solution
  auto solution = [](double x, double y) { return x * x - y * y; };

  // Discretize PDE
  auto alpha = [](double, double) { return 1.0; };
  solver::FiniteDifference2D fd(alpha, mesh, bc, source);
  fd.discretize();
  fd.updateRHS();

  // A: interior Laplacian matrix
  // b: boundary contribution from Dirichlet nodes
  const auto& A = fd.getMatrixK();
  Eigen::VectorXd b = fd.getVector(), exact(A.cols());

  ASSERT_EQ(A.cols(), mesh.getInnerNodes().size());
  ASSERT_EQ(b.size(), A.rows());

  std::size_t j = 0;
  auto nodes = mesh.getNodes();
  for (std::size_t nodeID : mesh.getInnerNodes())
    exact[j++] = solution(nodes[nodeID].x_, nodes[nodeID].y_);

  // Discrete residual should vanish up to roundoff
  Eigen::VectorXd res = A * exact + b;
  EXPECT_LT(res.lpNorm<Eigen::Infinity>(), 1e-12);
}

// =============================================================================
// Test 3 - Verify the expected convergence rate (2nd-order) to solve the
//          Laplace equation with Dirichlet BCs in all sides.
//
// For u(x,y) = sin(πx/Lx) * sinh(πy/Lx), -div(α∇u) = 0. Imposing Dirichlet BCs
// at all sides leads to u_left = u_right = u_bottom = 0, u_top = sin(πx/Lx) *
// sinh(πLy/Lx).
//
// Two different mesh sizes are used to test convergence, with
// h_fine = 0.5 * h_coarse.
// =============================================================================
TEST(FiniteDifference2D, LaplaceDirichletBCconvergence) {
  constexpr std::size_t n_coarse = 51;
  constexpr std::size_t n_fine = 101;
  constexpr double Lx = 1, Ly = 1;

  const mesh::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, n_coarse, n_coarse);
  const mesh::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, n_fine, n_fine);

  // Define BCs
  auto zeroBC = [](double x, double y, double t) { return 0; };
  auto topBC = [&](double x, double y, double t) {
    return std::sin(M_PI * x / Lx) * std::sinh(M_PI * Ly / Lx);
  };

  solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(topBC);

  // Source term
  auto source = [](double, double, double) { return 0.0; };

  // Exact solution
  auto solution = [&](double x, double y) {
    return std::sin(M_PI * x / Lx) * std::sinh(M_PI * y / Lx);
  };

  // Discretize PDE
  auto alpha = [](double, double) { return 1.0; };
  solver::FiniteDifference2D fd_coarse(alpha, mesh_coarse, bc, source);
  solver::FiniteDifference2D fd_fine(alpha, mesh_fine, bc, source);

  double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
  double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
  double h_coarse = mesh_coarse.getDx();
  double h_fine = mesh_fine.getDx();

  // Verify expected convergence rate
  double convergence_rate =
      std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
  EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test 4 - Verify the expected convergence rate (2nd-order) to solve the
//          Laplace equation with mixed BCs.
//
// For u(x,y) = sin(πx/Lx) * cosh(πy/Lx), --div(α∇u) = 0. Imposing Dirichlet BCs
// on the right, left, and bottom sides leads to u_left = u_right = 0, and
// u_bottom = (πx/Lx). Imposing a Neumann BC on the top reads du/dy|_top =
// = π/Lx * sin(πx/Lx) * sinh(πy/Lx)
//
// Two different mesh sizes are used to test convergence, with
// h_fine = 0.5 * h_coarse.
// =============================================================================
TEST(FiniteDifference2D, LaplaceMixedBCconvergence) {
  constexpr std::size_t n_coarse = 51;
  constexpr std::size_t n_fine = 101;
  constexpr double Lx = 2.0, Ly = 3.0;

  const mesh::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, n_coarse, n_coarse);
  const mesh::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, n_fine, n_fine);

  // Define BCs
  auto zeroBC = [](double, double, double) { return 0.0; };
  auto bottomBC = [&](double x, double, double) {
    return std::sin(M_PI * x / Lx);
  };
  auto topBC = [&](double x, double, double) {
    return M_PI / Lx * std::sin(M_PI * x / Lx) * std::sinh(M_PI * Ly / Lx);
  };

  solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(bottomBC);
  bc["Top"] = std::make_shared<bc::NeumannBoundaryCondition>(topBC);

  // Source term
  auto source = [](double, double, double) { return 0.0; };

  // Exact solution
  auto solution = [&](double x, double y) {
    return std::sin(M_PI * x / Lx) * std::cosh(M_PI * y / Lx);
  };

  // Discretize PDE
  auto alpha = [](double, double) { return 1.0; };
  solver::FiniteDifference2D fd_coarse(alpha, mesh_coarse, bc, source);
  solver::FiniteDifference2D fd_fine(alpha, mesh_fine, bc, source);

  double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
  double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
  double h_coarse = mesh_coarse.getDx();
  double h_fine = mesh_fine.getDx();

  // Verify expected convergence rate
  double convergence_rate =
      std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
  EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test 5 - Verify that the Laplacian matrix with pure Neumann BCs has a
//          constant vector in its nullspace. That is, A * ones = 0.
// =============================================================================
TEST(FiniteDifference2D, LaplaceNullSpace) {
  constexpr std::size_t n = 101;
  constexpr double Lx = 2.0, Ly = 3.0;
  mesh::StructuredMesh2D mesh(0, Lx, 0, Ly, n, n);

  // Define BCs
  auto zeroBC = [](double, double, double) { return 0.0; };
  solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);
  bc["Top"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);

  // Source term
  auto source = [](double, double, double) { return 0.0; };

  // Discretize PDE
  auto alpha = [](double, double) { return 1.0; };
  solver::FiniteDifference2D fd(alpha, mesh, bc, source);
  fd.discretize();

  const auto& A = fd.getMatrixK();
  Eigen::VectorXd ones = Eigen::VectorXd::Constant(A.cols(), 1.0);

  EXPECT_NEAR((A * ones).lpNorm<Eigen::Infinity>(), 0.0, 1e-12);
}

// =============================================================================
// Test 6 - Verify the expected convergence rate (2nd-order) to solve the
//          Poisson equation with mixed BCs.
//
// For u(x,y) = exp(-x²) * sin(πy), -div(α∇u) = -(4x² - 2 - π²) * exp(-x²) *
// * sin(πy). Imposing Dirichlet BCs on the bottom and top sides leads to
// u_bottom = u_top = 0. Imposing Neumann BCs on the left and right reads
// du/dy|_left = 0, and du/dy|_right = -2 / exp(1) * sin(πy).
//
// Two different mesh sizes are used to test convergence, with
// h_fine = 0.5 * h_coarse.
// =============================================================================
TEST(FiniteDifference2D, PoissonMixedBCconvergence) {
  constexpr std::size_t n_coarse = 51;
  constexpr std::size_t n_fine = 101;

  const mesh::StructuredMesh2D mesh_coarse(0, 1, 0, 1, n_coarse, n_coarse);
  const mesh::StructuredMesh2D mesh_fine(0, 1, 0, 1, n_fine, n_fine);

  // Define BCs
  auto zeroBC = [](double, double, double) { return 0.0; };
  auto rightBC = [](double, double y, double) {
    return -2 * std::exp(-1) * std::sin(M_PI * y);
  };

  solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::NeumannBoundaryCondition>(rightBC);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);

  // Exact solution
  auto solution = [](double x, double y) {
    return std::exp(-x * x) * std::sin(M_PI * y);
  };

  // Source term:
  auto source = [&](double x, double y, double) {
    return -(4 * x * x - 2 - M_PI * M_PI) * solution(x, y);
  };

  // Discretize PDE
  auto alpha = [](double, double) { return 1.0; };
  solver::FiniteDifference2D fd_coarse(alpha, mesh_coarse, bc, source);
  solver::FiniteDifference2D fd_fine(alpha, mesh_fine, bc, source);

  double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
  double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
  double h_coarse = mesh_coarse.getDx();
  double h_fine = mesh_fine.getDx();

  double convergence_rate =
      std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
  EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test 7 - Verify the Poisson equation is solved correctly with all three BC
//          types present simultaneously (Dirichlet, Neumann, Robin):
//             (Left, Bottom)  = Dirichlet-Neumann
//             (Bottom, Right) = Neumann-Robin
//             (Right, Top)    = Robin-Dirichlet
//             (Top, Left)     = Dirichlet-Dirichlet
//
// Manufactured solution: u(x,y) = log(sin²(xy) + 1), with
// -div(α∇u) = -(x²+y²)(3cos(2xy)-1) / (1+sin²(xy))², α = 1.
//
// BCs:
//   Left   (x=0):   Dirichlet, u = 0
//   Bottom (y=0):   Neumann,   du/dy = 0
//   Right  (x=Lx):  Robin,     2*u + du/dx = f_right(y)
//   Top    (y=Ly):  Dirichlet, u = log(sin²(x) + 1)
// =============================================================================
TEST(FiniteDifference2D, PoissonAllBCTypesCornerCombinations) {
  constexpr std::size_t nx = 101, ny = 51;
  constexpr double Lx = 2.0, Ly = 1.0;
  const mesh::StructuredMesh2D mesh(0, Lx, 0, Ly, nx, ny);

  // Exact solution
  auto exact = [](double x, double y) {
    double s = std::sin(x * y);
    return std::log(s * s + 1);
  };

  // Define BCs
  solver::BoundaryConditions bc;
  auto dudx = [](double x, double y) {
    double s = std::sin(x * y);
    return y * std::sin(2 * x * y) / (1.0 + s * s);
  };

  auto zeroBC = [](double, double, double) { return 0.0; };
  auto uCoeffRight = [](double, double, double) { return 2.0; };
  auto duCoeffRight = [](double, double, double) { return 1.0; };
  auto fRight = [&](double, double y, double) {
    return 2.0 * exact(Lx, y) + dudx(Lx, y);
  };
  auto topBC = [&](double x, double, double) { return exact(x, Ly); };

  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::RobinBoundaryCondition>(
      uCoeffRight, duCoeffRight, fRight);
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(topBC);

  // Source term
  auto source = [](double x, double y, double) {
    double s = std::sin(x * y);
    double d = 1.0 + s * s;
    return -(x * x + y * y) * (3.0 * std::cos(2 * x * y) - 1.0) / (d * d);
  };

  auto alpha = [](double, double) { return 1.0; };

  // Discretize PDE
  solver::FiniteDifference2D fd(alpha, mesh, bc, source);

  double err = solve_and_get_error(fd, mesh, exact);

  EXPECT_LT(err, 1e-3);
}

// =============================================================================
// Test 8 - Verify the expected convergence rate (2nd-order) to solve the
//          Poisson equation with a source and variable diffusivity.
//
// For u(x,y) = sin(πx)sin(πy) and α(x,y) = 1 + x, -div(α∇u) = πcos(πx)sin(πy)
// - 2π²(1+x)sin(πx)sin(πy). Imposing Dirichlet BCs at all four sides leads to
// u_left = u_right = u_bottom = u_top = 0.
//
// With the choice of parameters, O(error) ≈ O(dx²) ≈ 4e-4 < 1e-3.
// =============================================================================
TEST(FiniteDifference2D, PoissonVariableAlphaConvergence) {
  constexpr std::size_t n_coarse = 51, n_fine = 101;

  const mesh::StructuredMesh2D mesh_coarse(0, 1, 0, 1, n_coarse, n_coarse);
  const mesh::StructuredMesh2D mesh_fine(0, 1, 0, 1, n_fine, n_fine);

  // Define BCs
  auto zero = [](double, double, double) { return 0.0; };
  solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zero);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zero);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(zero);
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(zero);

  // Source term
  auto source = [](double x, double y, double) {
    return -(M_PI * std::cos(M_PI * x) * std::sin(M_PI * y) -
             2.0 * M_PI * M_PI * (1.0 + x) * std::sin(M_PI * x) *
                 std::sin(M_PI * y));
  };

  // Exact solution
  auto solution = [](double x, double y) {
    return std::sin(M_PI * x) * std::sin(M_PI * y);
  };

  // Discretize PDE
  auto alpha = [](double x, double) { return 1.0 + x; };
  solver::FiniteDifference2D fd_coarse(alpha, mesh_coarse, bc, source);
  solver::FiniteDifference2D fd_fine(alpha, mesh_fine, bc, source);

  // Verify expected convergence rate
  double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
  double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);

  double rate = std::log(err_coarse / err_fine) /
                std::log(mesh_coarse.getDx() / mesh_fine.getDx());

  EXPECT_NEAR(rate, 2.0, 0.1);
}

// =============================================================================
// Test 9 - Verify that Robin BCs reduce to Neumann BCs when u_coeff = 0.
//
// For the same exact problem, changing Neumann BCs to Robin BCs with
// u_coeff = 0 should yield the same solution.
// =============================================================================
TEST(FiniteDifference2D, RobinReducesToNeumannWhenUCoeffZero) {
  constexpr std::size_t nx = 21, ny = 21;
  constexpr double Lx = 2.0, Ly = 3.0;
  const mesh::StructuredMesh2D mesh(0, Lx, 0, Ly, nx, ny);

  // Define BCs
  auto zeroBC = [](double, double, double) { return 0.0; };
  auto bottomBC = [&](double x, double, double) {
    return std::sin(M_PI * x / Lx);
  };
  auto topFluxNeumann = [&](double x, double, double) {
    return M_PI / Lx * std::sin(M_PI * x / Lx) * std::sinh(M_PI * Ly / Lx);
  };
  auto duCoeff = [](double, double, double) { return 2.0; };
  auto uCoeff0 = [](double, double, double) { return 0.0; };
  auto topFluxRobin = [&](double x, double y, double t) {
    return duCoeff(x, y, t) * topFluxNeumann(x, y, t);
  };

  solver::BoundaryConditions bc_neumann;
  bc_neumann["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc_neumann["Right"] =
      std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc_neumann["Bottom"] =
      std::make_shared<bc::DirichletBoundaryCondition>(bottomBC);
  bc_neumann["Top"] =
      std::make_shared<bc::NeumannBoundaryCondition>(topFluxNeumann);

  solver::BoundaryConditions bc_robin;
  bc_robin["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc_robin["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc_robin["Bottom"] =
      std::make_shared<bc::DirichletBoundaryCondition>(bottomBC);
  bc_robin["Top"] = std::make_shared<bc::RobinBoundaryCondition>(
      uCoeff0, duCoeff, topFluxRobin);

  // Source term
  auto source = [](double, double, double) { return 0.0; };

  // Exact solution
  auto alpha = [](double, double) { return 1.0; };

  // Discretize PDE
  solver::FiniteDifference2D fd_neumann(alpha, mesh, bc_neumann, source);
  solver::FiniteDifference2D fd_robin(alpha, mesh, bc_robin, source);

  // Solve both problems
  fd_neumann.discretize();
  fd_robin.discretize();

  Eigen::VectorXd sol_neumann = fd_neumann.solveSteadyState();
  Eigen::VectorXd sol_robin = fd_robin.solveSteadyState();

  EXPECT_LT((sol_neumann - sol_robin).lpNorm<Eigen::Infinity>(), 1e-12);
}

// =============================================================================
// Test 10 - Verify 2nd-order convergence for the Laplace equation with a true
//           Robin BC (u_coeff, du_coeff both non-trivial) on the top boundary.
//
// u(x,y) = sin(πx/Lx) * cosh(πy/Lx) is harmonic. Dirichlet on
// left/right/bottom: u_left = u_right = 0, u_bottom = sin(πx/Lx). On top, with
// u_coeff = 2 and du_coeff = 3, we have f(x) = 2 * u_top + 3 * du/dy|_top
//          = sin(πx/Lx) * [2 cosh(πLy/Lx) + 3(π/Lx) sinh(πLy/Lx)]
// =============================================================================
TEST(FiniteDifference2D, LaplaceRobinBCConvergence) {
  constexpr std::size_t n_coarse = 51;
  constexpr std::size_t n_fine = 101;
  constexpr double Lx = 2.0, Ly = 3.0;

  const mesh::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, n_coarse, n_coarse);
  const mesh::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, n_fine, n_fine);

  // Define BCs
  auto zeroBC = [](double, double, double) { return 0.0; };
  auto bottomBC = [&](double x, double, double) {
    return std::sin(M_PI * x / Lx);
  };

  auto uCoeff = [](double, double, double) { return 2.0; };
  auto duCoeff = [](double, double, double) { return 3.0; };
  auto topRobin = [&](double x, double, double) {
    return std::sin(M_PI * x / Lx) *
           (2.0 * std::cosh(M_PI * Ly / Lx) +
            3.0 * (M_PI / Lx) * std::sinh(M_PI * Ly / Lx));
  };

  solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(bottomBC);
  bc["Top"] =
      std::make_shared<bc::RobinBoundaryCondition>(uCoeff, duCoeff, topRobin);

  // Source term
  auto source = [](double, double, double) { return 0.0; };

  // Exact solution
  auto solution = [&](double x, double y) {
    return std::sin(M_PI * x / Lx) * std::cosh(M_PI * y / Lx);
  };

  auto alpha = [](double, double) { return 1.0; };

  // Discretize PDE
  solver::FiniteDifference2D fd_coarse(alpha, mesh_coarse, bc, source);
  solver::FiniteDifference2D fd_fine(alpha, mesh_fine, bc, source);

  double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, solution);
  double err_fine = solve_and_get_error(fd_fine, mesh_fine, solution);
  double h_coarse = mesh_coarse.getDx();
  double h_fine = mesh_fine.getDx();

  // Verify expected convergence rate
  double convergence_rate =
      std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
  EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test 11 - Verify the expected convergence rate (2nd-order) to solve the
//           Poisson equation with all three BC types present simultaneously.
//
// Same manufactured solution as Test 7: u(x,y) = log(sin²(xy) + 1), with
// -div(α∇u) = -(x²+y²)(3cos(2xy)-1) / (1+sin²(xy))², α = 1.
//
// BCs:
//   Left   (x=0):   Dirichlet, u = 0
//   Bottom (y=0):   Neumann,   du/dy = 0
//   Right  (x=Lx):  Robin,     2*u + 1*(du/dx) = f_right(y)
//   Top    (y=Ly):  Dirichlet, u = log(sin²(x) + 1)
// =============================================================================
TEST(FiniteDifference2D, PoissonAllBCTypesCornerCombinationsConvergence) {
  constexpr double Lx = 2.0, Ly = 1.0;
  constexpr std::size_t nx_coarse = 51, ny_coarse = 26;
  constexpr std::size_t nx_fine = 101, ny_fine = 51;
  const mesh::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, nx_coarse, ny_coarse);
  const mesh::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, nx_fine, ny_fine);

  // Exact solution
  auto exact = [](double x, double y) {
    double s = std::sin(x * y);
    return std::log(s * s + 1);
  };

  // Define BCs
  solver::BoundaryConditions bc;
  auto zeroBC = [](double, double, double) { return 0.0; };

  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);

  // du/dx for Robin BC
  auto dudx = [](double x, double y) {
    double s = std::sin(x * y);
    return y * std::sin(2 * x * y) / (1.0 + s * s);
  };

  auto uCoeffRight = [](double, double, double) { return 2.0; };
  auto duCoeffRight = [](double, double, double) { return 1.0; };
  auto fRight = [&](double, double y, double) {
    return 2.0 * exact(Lx, y) + dudx(Lx, y);
  };
  bc["Right"] = std::make_shared<bc::RobinBoundaryCondition>(
      uCoeffRight, duCoeffRight, fRight);

  auto topBC = [&](double x, double, double) { return exact(x, Ly); };
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(topBC);

  // Source term
  auto source = [](double x, double y, double) {
    double s = std::sin(x * y);
    double d = 1.0 + s * s;
    return -(x * x + y * y) * (3.0 * std::cos(2 * x * y) - 1.0) / (d * d);
  };

  auto alpha = [](double, double) { return 1.0; };

  // Discretize PDE
  solver::FiniteDifference2D fd_coarse(alpha, mesh_coarse, bc, source);
  solver::FiniteDifference2D fd_fine(alpha, mesh_fine, bc, source);

  double err_coarse = solve_and_get_error(fd_coarse, mesh_coarse, exact);
  double err_fine = solve_and_get_error(fd_fine, mesh_fine, exact);
  double h_coarse = mesh_coarse.getDx();
  double h_fine = mesh_fine.getDx();

  // Verify expected convergence rate
  double convergence_rate =
      std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
  EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// IsoparametricMapping tests
// =============================================================================

// =============================================================================
// Test 1 - toPhysical maps the reference vertices to the element nodes
// =============================================================================
TEST(IsoparametricMapping, MapsReferenceVertices) {
  mesh::Node2D a{0, 0.0, 0.0};
  mesh::Node2D b{1, 1.0, 0.0};
  mesh::Node2D c{2, 0.0, 1.0};
  heat2d::solver::IsoparametricMapping mapping({a, b, c});

  EXPECT_NEAR(mapping.toPhysical(0.0, 0.0).x_, 0.0, 1e-12);
  EXPECT_NEAR(mapping.toPhysical(0.0, 0.0).y_, 0.0, 1e-12);
  EXPECT_NEAR(mapping.toPhysical(1.0, 0.0).x_, 1.0, 1e-12);
  EXPECT_NEAR(mapping.toPhysical(1.0, 0.0).y_, 0.0, 1e-12);
  EXPECT_NEAR(mapping.toPhysical(0.0, 1.0).x_, 0.0, 1e-12);
  EXPECT_NEAR(mapping.toPhysical(0.0, 1.0).y_, 1.0, 1e-12);
}

// =============================================================================
// Test 2 - The Jacobian determinant equals twice the physical element area
// =============================================================================
TEST(IsoparametricMapping, DetJacobianEqualsTwiceArea) {
  mesh::Node2D a{0, 0.0, 0.0};
  mesh::Node2D b{1, 2.0, 1.0};
  mesh::Node2D c{2, -1.0, 3.0};
  heat2d::solver::IsoparametricMapping mapping({a, b, c});

  double area = 3.5;
  EXPECT_NEAR(mapping.detJacobian(0.3, 0.2), 2.0 * area, 1e-12);
}

// =============================================================================
// Test 3 - An affine triangle has constant physical gradients
// =============================================================================
TEST(IsoparametricMapping, AffineElementConstantGradients) {
  mesh::Node2D a{0, 0.0, 0.0};
  mesh::Node2D b{1, 2.0, 0.0};
  mesh::Node2D c{2, 0.0, 3.0};
  heat2d::solver::IsoparametricMapping mapping({a, b, c});

  Eigen::Matrix<double, 3, 2> g1 = mapping.physicalGradients(0.1, 0.4);
  Eigen::Matrix<double, 3, 2> g2 = mapping.physicalGradients(0.9, 0.1);

  Eigen::Matrix<double, 3, 2> expected;
  expected << -0.5, -1.0 / 3.0,  //
      0.5, 0.0,                  //
      0.0, 1.0 / 3.0;
  EXPECT_TRUE(g1.isApprox(expected, 1e-12));
  EXPECT_TRUE((g1 - g2).norm() < 1e-12);
}

// =============================================================================
// FiniteElement2D tests
// =============================================================================

// =============================================================================
// Helper: L-infinity error of the FE solution against an exact function
// =============================================================================
double fe_solve_and_get_error(heat2d::solver::SpatialDiscretization2D& sd,
                              const heat2d::mesh::Mesh2D& mesh,
                              std::function<double(double, double)> solution) {
  sd.discretize();
  Eigen::VectorXd sol = sd.solveSteadyState();
  Eigen::VectorXd exact(sol.size());

  std::size_t j = 0;
  for (const auto& node : mesh.getNodes())
    exact[j++] = solution(node.x_, node.y_);

  return (exact - sol).lpNorm<Eigen::Infinity>();
}

// =============================================================================
// Test - Patch test: a linear field is reproduced exactly with pure Dirichlet
// =============================================================================
TEST(FiniteElement2D, PatchTestLinearField) {
  constexpr std::size_t nx = 5, ny = 5;
  const heat2d::mesh::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);

  auto linearBC = [](double x, double y, double) {
    return 1.0 + 2.0 * x - 0.5 * y;
  };
  heat2d::solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(linearBC);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(linearBC);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(linearBC);
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(linearBC);

  auto source = [](double, double, double) { return 0.0; };
  auto alpha = [](double, double) { return 1.0; };

  auto exact = [](double x, double y) { return 1.0 + 2.0 * x - 0.5 * y; };

  heat2d::solver::FiniteElement2D fe(alpha, mesh, bc, source, 2);
  EXPECT_LT(fe_solve_and_get_error(fe, mesh, exact), 1e-9);
}

// =============================================================================
// Test - Patch test: a linear field is reproduced exactly with mixed
//        Dirichlet/Neumann BCs (u = x, Dirichlet on the left, flux on the
//        remaining sides).
// =============================================================================
TEST(FiniteElement2D, MixedDirichletNeumannPatchTest) {
  constexpr std::size_t nx = 9, ny = 9;
  const heat2d::mesh::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);

  auto leftBC = [](double, double, double) { return 0.0; };
  auto zeroFlux = [](double, double, double) { return 0.0; };
  auto unitFlux = [](double, double, double) { return 1.0; };

  heat2d::solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(leftBC);
  bc["Bottom"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroFlux);
  bc["Top"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroFlux);
  bc["Right"] = std::make_shared<bc::NeumannBoundaryCondition>(unitFlux);

  auto source = [](double, double, double) { return 0.0; };
  auto alpha = [](double, double) { return 1.0; };

  auto exact = [](double x, double y) { return x; };

  heat2d::solver::FiniteElement2D fe(alpha, mesh, bc, source, 3);
  EXPECT_LT(fe_solve_and_get_error(fe, mesh, exact), 1e-9);
}

// =============================================================================
// Test - Dirichlet-only Poisson problem converges with order 2
// =============================================================================
TEST(FiniteElement2D, PoissonConvergence) {
  auto exact = [](double x, double y) {
    return std::sin(M_PI * x) * std::sin(M_PI * y);
  };
  auto source = [](double x, double y, double) {
    return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
  };
  auto zeroBC = [](double, double, double) { return 0.0; };

  heat2d::solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);

  auto alpha = [](double, double) { return 1.0; };

  const heat2d::mesh::StructuredMesh2D mesh_coarse(0, 1, 0, 1, 9, 9);
  const heat2d::mesh::StructuredMesh2D mesh_fine(0, 1, 0, 1, 17, 17);

  heat2d::solver::FiniteElement2D fe_coarse(alpha, mesh_coarse, bc, source, 3);
  heat2d::solver::FiniteElement2D fe_fine(alpha, mesh_fine, bc, source, 3);

  double err_coarse = fe_solve_and_get_error(fe_coarse, mesh_coarse, exact);
  double err_fine = fe_solve_and_get_error(fe_fine, mesh_fine, exact);
  double h_coarse = mesh_coarse.getDx();
  double h_fine = mesh_fine.getDx();

  double convergence_rate =
      std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
  EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test - Poisson equation with a source and variable diffusivity α = 1 + x
//        converges with order 2.
// =============================================================================
TEST(FiniteElement2D, PoissonVariableAlphaConvergence) {
  constexpr std::size_t n_coarse = 17, n_fine = 33;

  const heat2d::mesh::StructuredMesh2D mesh_coarse(0, 1, 0, 1, n_coarse,
                                                   n_coarse);
  const heat2d::mesh::StructuredMesh2D mesh_fine(0, 1, 0, 1, n_fine, n_fine);

  // Define BCs
  auto zero = [](double, double, double) { return 0.0; };
  heat2d::solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zero);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zero);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(zero);
  bc["Top"] = std::make_shared<bc::DirichletBoundaryCondition>(zero);

  // Source term
  auto source = [](double x, double y, double) {
    return -(M_PI * std::cos(M_PI * x) * std::sin(M_PI * y) -
             2.0 * M_PI * M_PI * (1.0 + x) * std::sin(M_PI * x) *
                 std::sin(M_PI * y));
  };

  // Exact solution
  auto solution = [](double x, double y) {
    return std::sin(M_PI * x) * std::sin(M_PI * y);
  };

  auto alpha = [](double x, double) { return 1.0 + x; };
  heat2d::solver::FiniteElement2D fe_coarse(alpha, mesh_coarse, bc, source, 3);
  heat2d::solver::FiniteElement2D fe_fine(alpha, mesh_fine, bc, source, 3);

  double err_coarse = fe_solve_and_get_error(fe_coarse, mesh_coarse, solution);
  double err_fine = fe_solve_and_get_error(fe_fine, mesh_fine, solution);
  double h_coarse = mesh_coarse.getDx();
  double h_fine = mesh_fine.getDx();

  double convergence_rate =
      std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
  EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test - Laplace equation with a true Robin BC on the top boundary converges
//        with order 2.
//
// u(x,y) = sin(πx/Lx) * cosh(πy/Lx) is harmonic. Dirichlet on
// left/right/bottom: u_left = u_right = 0, u_bottom = sin(πx/Lx). On top, with
// u_coeff = 2 and du_coeff = 3, we have f(x) = 2 * u_top + 3 * du/dy|_top
//          = sin(πx/Lx) * [2 cosh(πLy/Lx) + 3(π/Lx) sinh(πLy/Lx)]
// =============================================================================
TEST(FiniteElement2D, LaplaceRobinBCConvergence) {
  constexpr std::size_t n_coarse = 17, n_fine = 33;
  constexpr double Lx = 2.0, Ly = 3.0;

  const heat2d::mesh::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, n_coarse,
                                                   n_coarse);
  const heat2d::mesh::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, n_fine, n_fine);

  auto zeroBC = [](double, double, double) { return 0.0; };
  auto bottomBC = [&](double x, double, double) {
    return std::sin(M_PI * x / Lx);
  };

  auto uCoeff = [](double, double, double) { return 2.0; };
  auto duCoeff = [](double, double, double) { return 3.0; };
  auto topRobin = [&](double x, double, double) {
    return std::sin(M_PI * x / Lx) *
           (2.0 * std::cosh(M_PI * Ly / Lx) +
            3.0 * (M_PI / Lx) * std::sinh(M_PI * Ly / Lx));
  };

  heat2d::solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::DirichletBoundaryCondition>(bottomBC);
  bc["Top"] =
      std::make_shared<bc::RobinBoundaryCondition>(uCoeff, duCoeff, topRobin);

  auto source = [](double, double, double) { return 0.0; };
  auto solution = [&](double x, double y) {
    return std::sin(M_PI * x / Lx) * std::cosh(M_PI * y / Lx);
  };
  auto alpha = [](double, double) { return 1.0; };

  heat2d::solver::FiniteElement2D fe_coarse(alpha, mesh_coarse, bc, source, 3);
  heat2d::solver::FiniteElement2D fe_fine(alpha, mesh_fine, bc, source, 3);

  double err_coarse = fe_solve_and_get_error(fe_coarse, mesh_coarse, solution);
  double err_fine = fe_solve_and_get_error(fe_fine, mesh_fine, solution);
  double h_coarse = mesh_coarse.getDx();
  double h_fine = mesh_fine.getDx();

  double convergence_rate =
      std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
  EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test - Poisson equation with all three BC types present simultaneously
//        (Dirichlet, Neumann, Robin) converges with order 2.
//
// Manufactured solution: u(x,y) = log(sin²(xy) + 1) with α = 1.
//   Left   (x=0):   Dirichlet, u = 0
//   Bottom (y=0):   Neumann,   du/dy = 0
//   Right  (x=Lx):  Dirichlet, u = log(sin²(Lx·y) + 1)
//   Top    (y=Ly):  Robin,     2*u + 1*(du/dy) = f_top(x)
// =============================================================================
// =============================================================================
// Test - Poisson equation with all three BC types present simultaneously
//        (Dirichlet, Neumann, Robin) converges with order 2.
//
// Manufactured solution: u(x,y) = sin(pi x/Lx) sin(pi y/Ly) with alpha = 1.
//   Left   (x=0):   Dirichlet, u = 0
//   Right  (x=Lx):  Dirichlet, u = 0
//   Bottom (y=0):   Neumann,   q = alpha du/dn = -(pi/Ly) sin(pi x/Lx)
//   Top    (y=Ly):  Robin,     2*u + 1*(du/dy) = -(pi/Ly) sin(pi x/Lx)
// =============================================================================
TEST(FiniteElement2D, PoissonAllBCTypesCornerCombinationsConvergence) {
  constexpr double Lx = 2.0, Ly = 1.0;
  constexpr std::size_t nx_coarse = 33, ny_coarse = 17;
  constexpr std::size_t nx_fine = 65, ny_fine = 33;
  const heat2d::mesh::StructuredMesh2D mesh_coarse(0, Lx, 0, Ly, nx_coarse,
                                                   ny_coarse);
  const heat2d::mesh::StructuredMesh2D mesh_fine(0, Lx, 0, Ly, nx_fine,
                                                 ny_fine);

  auto exact = [](double x, double y) {
    return std::sin(M_PI * x / Lx) * std::sin(M_PI * y / Ly);
  };
  auto source = [](double x, double y, double) {
    double mu = M_PI * M_PI / (Lx * Lx) + M_PI * M_PI / (Ly * Ly);
    return mu * std::sin(M_PI * x / Lx) * std::sin(M_PI * y / Ly);
  };
  auto alpha = [](double, double) { return 1.0; };
  auto zeroBC = [](double, double, double) { return 0.0; };

  // Bottom Neumann: q = alpha du/dn = -(pi/Ly) sin(pi x/Lx).
  auto bottomFlux = [](double x, double, double) {
    return -M_PI / Ly * std::sin(M_PI * x / Lx);
  };
  // Top Robin: 2u + du/dy = f, du/dy at y = Ly = -(pi/Ly) sin(pi x/Lx).
  auto uCoeffTop = [](double, double, double) { return 2.0; };
  auto duCoeffTop = [](double, double, double) { return 1.0; };
  auto fTop = [](double x, double, double) {
    return -M_PI / Ly * std::sin(M_PI * x / Lx);
  };

  heat2d::solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Right"] = std::make_shared<bc::DirichletBoundaryCondition>(zeroBC);
  bc["Bottom"] = std::make_shared<bc::NeumannBoundaryCondition>(bottomFlux);
  bc["Top"] =
      std::make_shared<bc::RobinBoundaryCondition>(uCoeffTop, duCoeffTop, fTop);

  heat2d::solver::FiniteElement2D fe_coarse(alpha, mesh_coarse, bc, source, 3);
  heat2d::solver::FiniteElement2D fe_fine(alpha, mesh_fine, bc, source, 3);

  double err_coarse = fe_solve_and_get_error(fe_coarse, mesh_coarse, exact);
  double err_fine = fe_solve_and_get_error(fe_fine, mesh_fine, exact);
  double h_coarse = mesh_coarse.getDx();
  double h_fine = mesh_fine.getDx();

  double convergence_rate =
      std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
  EXPECT_NEAR(convergence_rate, 2.0, 0.1);
}

// =============================================================================
// Test - The mass matrix must NOT contain the diffusivity: assembling with two
//        different α fields yields the same mass matrix.
// =============================================================================
TEST(FiniteElement2D, MassMatrixHasNoDiffusivity) {
  constexpr std::size_t nx = 9, ny = 9;
  const heat2d::mesh::StructuredMesh2D mesh(0, 1, 0, 1, nx, ny);

  auto zeroFlux = [](double, double, double) { return 0.0; };
  heat2d::solver::BoundaryConditions bc;
  bc["Left"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroFlux);
  bc["Right"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroFlux);
  bc["Bottom"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroFlux);
  bc["Top"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroFlux);

  auto source = [](double, double, double) { return 0.0; };
  auto alpha_one = [](double, double) { return 1.0; };
  auto alpha_x = [](double x, double) { return 1.0 + x; };

  heat2d::solver::FiniteElement2D fe_one(alpha_one, mesh, bc, source, 2);
  fe_one.discretize();
  heat2d::solver::FiniteElement2D fe_x(alpha_x, mesh, bc, source, 2);
  fe_x.discretize();

  const auto& M_one = fe_one.getMatrixM();
  const auto& M_x = fe_x.getMatrixM();
  EXPECT_EQ(M_one.rows(), M_x.rows());
  EXPECT_EQ(M_one.cols(), M_x.cols());
  EXPECT_LT((M_one - M_x).norm(), 1e-12);

  Eigen::VectorXd rowSums = M_one * Eigen::VectorXd::Ones(M_one.cols());
  EXPECT_GT(rowSums.minCoeff(), 0.0);
}
