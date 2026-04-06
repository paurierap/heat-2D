#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <vector>

#include "HeatPDE2D.hpp"
#include "DirichletBoundaryCondition.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "FiniteDifference2D.hpp"
#include "StructuredMesh2D.hpp"
#include "ExplicitEuler.hpp"
#include "ImplicitEuler.hpp"
#include "CrankNicolson.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Helper: solves the heat equation and returns the solution as a vectorised 
//         Eigen::VectorXd
// =============================================================================
Eigen::VectorXd solve_and_get_solution(HeatPDE2D& solver, double t_end)
{
    solver.integrate(t_end);
    return solver.getSolution();
}

// =============================================================================
// Helper: compares approximation with exact solution
// =============================================================================
double solve_and_get_error(HeatPDE2D& solver, const spatial::Mesh2D& mesh, std::function<double (double, double, double)> solution, double t_end)
{
    Eigen::VectorXd sol = solve_and_get_solution(solver, t_end);
    Eigen::VectorXd exact(sol.size());

    int j = 0;
    for (const auto& node : mesh.getNodes()) exact[j++] = solution(node.x_, node.y_, t_end);

    return (sol - exact).lpNorm<Eigen::Infinity>();
}

// =============================================================================
// Helper: compares approximation with reference approximation (with a refined 
//         discretization)
// =============================================================================
double solve_and_get_error_vs_ref(HeatPDE2D& solver, const Eigen::Ref<const Eigen::VectorXd>& ref, double t_end)
{
    Eigen::VectorXd sol = solve_and_get_solution(solver, t_end);
    return (sol - ref).lpNorm<Eigen::Infinity>();
}

// =============================================================================
// Fixture - Used to verify the expected convergence rates of Explicit Euler, 
//           Implicit Euler and Crank-Nicolson for the heat equation with 
//           Dirichlet BCs in all sides. 
//
// For u(x,y,t) = exp(-2π²αt) * sin(πx) * sin(πy), ∂u/∂t = αΔu. Imposing 
// Dirichlet BCs at all sides leads to u_left = u_right = u_bottom = u_top = 0, 
// and the initial condition u0 = u(x,y,0) = sin(πx) * sin(πy).
//
// The solution is approximated from t = 0 to t = 0.1/α. The parameter α is free
// but t_end and dt need to be scaled with it.
// =============================================================================
class DirichletBCTimeConvergence : public testing::Test
{
    protected: 
        static constexpr double alpha = 0.5 / (M_PI * M_PI);
        static constexpr double t_start = 0.0;
        static constexpr double t_end = 0.1 / alpha;

        std::function<double(double, double, double)> zeroBC = [](double, double, double){return 0.0;};

        std::function<double(double, double)> u0 = [](double x, double y){return std::sin(M_PI * x) * std::sin(M_PI * y);};

        std::function<double(double, double, double)> source = [](double, double, double){return 0.0;};

        std::function<double(double, double, double)> exact = [](double x, double y, double t)
        {return std::exp(-2 * M_PI * M_PI * alpha * t) * std::sin(M_PI * x) * std::sin(M_PI * y);};

        spatial::BoundaryConditions bc;

        void SetUp() override
        {
            // BCs built in SetUp() since they use shared_ptr
            bc[spatial::DomainSide::Left] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
            bc[spatial::DomainSide::Right] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
            bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
            bc[spatial::DomainSide::Top] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
        }
};

// =============================================================================
// Test 1: Explicit Euler (1st order convergence)
// 
// The parameters are chosen to ensure stability. Numerical stability is ensured 
// if dt < dx²/(4α). Therefore, dt_coarse = 2e-4 < 6.25e-4 = (1/20)²/4.
//
// As opposed to comparing against an analytical solution, the approximation is 
// compared against another approximation with the same spatial discretization, 
// but employing Crank Nicolson time integration. This ensures that spatial 
// errors cancel out.
// =============================================================================
TEST_F(DirichletBCTimeConvergence, ExplicitEuler)
{
    constexpr int n = 21;
    const spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Discretise PDE
    spatial::FiniteDifference2D EEfd_coarse(alpha, mesh, bc, source);
    spatial::FiniteDifference2D EEfd_fine(alpha, mesh, bc, source);
    spatial::FiniteDifference2D CNfd_ref(alpha, mesh, bc, source);

    // Time integrators
    constexpr double dt_coarse = 0.0002 / alpha;
    constexpr double dt_fine = 0.0001 / alpha;
    temporal::ExplicitEuler EEti_coarse(dt_coarse), EEti_fine(dt_fine);
    temporal::CrankNicolson CNti_ref(1e-4);

    HeatPDE2D EEsolver_coarse(EEfd_coarse, EEti_coarse, t_start, u0);
    HeatPDE2D EEsolver_fine(EEfd_fine, EEti_fine, t_start, u0);
    HeatPDE2D CNsolver_ref(CNfd_ref, CNti_ref, t_start, u0);

    // Solve and compare approximations
    Eigen::VectorXd ref = solve_and_get_solution(CNsolver_ref, t_end);
    double EE_err_coarse = solve_and_get_error_vs_ref(EEsolver_coarse, ref, t_end);
    double EE_err_fine   = solve_and_get_error_vs_ref(EEsolver_fine, ref, t_end);

    // Verify Explicit Euler expected convergence rate
    double EE_rate = std::log(EE_err_coarse / EE_err_fine) / std::log(dt_coarse / dt_fine);
    EXPECT_NEAR(EE_rate, 1, 0.1);
}

// =============================================================================
// Test 2: Implicit Euler (1st order convergence)
// 
// The parameters are chosen to ensure time integration error dominates over 
// spatial discretisation error. Provided that error(space) = O(dx²) ≈ 4.4e-5, 
// error(time) = O(dt) ≈ 5e-3.
// =============================================================================
TEST_F(DirichletBCTimeConvergence, ImplicitEuler)
{
    constexpr int n = 151;
    const spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);
    
    // Discretise PDE
    spatial::FiniteDifference2D IEfd_coarse(alpha, mesh, bc, source);
    spatial::FiniteDifference2D IEfd_fine(alpha, mesh, bc, source);

    // Time integrators
    constexpr double dt_coarse = 0.01 / alpha;
    constexpr double dt_fine = 0.005 / alpha;
    temporal::ImplicitEuler IEti_coarse(dt_coarse), IEti_fine(dt_fine);

    // Create solver object
    HeatPDE2D IEsolver_coarse(IEfd_coarse, IEti_coarse, t_start, u0);
    HeatPDE2D IEsolver_fine(IEfd_fine, IEti_fine, t_start, u0);

    // Solve and compare approximations
    double IE_err_coarse = solve_and_get_error(IEsolver_coarse, mesh, exact, t_end);
    double IE_err_fine = solve_and_get_error(IEsolver_fine, mesh, exact, t_end);

    // Verify Implicit Euler expected convergence rate
    double IE_rate = std::log(IE_err_coarse / IE_err_fine) / std::log(dt_coarse / dt_fine);
    EXPECT_NEAR(IE_rate, 1, 0.1) << "IE_err_coarse: " << IE_err_coarse << "\nIE_err_fine: " << IE_err_fine << "\n";
}

// =============================================================================
// Test 3: Crank Nicolson (2nd order convergence)
// 
// The parameters are chosen to ensure time integration error dominates over 
// spatial discretisation error. Provided that error(space) = O(dx²) ≈ 4.4e-5, 
// error(time) = O(dt²) ≈ 1e-4.
// =============================================================================
TEST_F(DirichletBCTimeConvergence, CrankNicolson)
{
    constexpr int n = 151;
    const spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Discretise PDE
    spatial::FiniteDifference2D CNfd_coarse(alpha, mesh, bc, source);
    spatial::FiniteDifference2D CNfd_fine(alpha, mesh, bc, source);

    // Time integrators
    constexpr double dt_coarse = 0.02 / alpha;
    constexpr double dt_fine = 0.01 / alpha;
    temporal::CrankNicolson CNti_coarse(dt_coarse), CNti_fine(dt_fine);

    // Create solver object
    HeatPDE2D CNsolver_coarse(CNfd_coarse, CNti_coarse, t_start, u0);
    HeatPDE2D CNsolver_fine(CNfd_fine, CNti_fine, t_start, u0);

    // Compare approximations
    double CN_err_coarse = solve_and_get_error(CNsolver_coarse, mesh, exact, t_end);
    double CN_err_fine = solve_and_get_error(CNsolver_fine, mesh, exact, t_end);

    // Verify Crank Nicolson expected convergence rate
    double CN_rate = std::log(CN_err_coarse / CN_err_fine) / std::log(dt_coarse / dt_fine);
    EXPECT_NEAR(CN_rate, 2, 0.1) << "CN_err_coarse: " << CN_err_coarse << "\nCN_err_fine: " << CN_err_fine << "\n";
}

// =============================================================================
// Test 4 - Verify that Crank Nicolson has the expected residual error.
//
// For u(x,y,t) = exp(-2π²αt) * sin(πx) * sin(πy) + 1/2 * exp(-5π²αt) * sin(2πx) 
// * sin(πy), ∂u/∂t = αΔu.  Imposing Dirichlet BCs at all sides leads to u_left 
// = u_right = u_bottom = u_top = 0, and the initial condition u0 = u(x,y,0) = 
// = sin(πx) * sin(πy) + 1/2 * sin(2πx) * sin(πy) is used.
//
// The solution is approximated from t = 0 to t = 0.1/α. 
//
// With the choice of parameters, O(error) ≈ O(dx²) + O(dt²) ≈ 5e-4 < 1e-3.
// =============================================================================
TEST(HeatPDE2D, CrankNicolsonExpectedError)
{
    constexpr int n = 101;
    constexpr double alpha = 0.5 / (M_PI * M_PI);
    const spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Boundary conditions
    auto zeroBC = [](double, double, double){return 0.0;};
    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Top] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);

    // Source function
    auto source = [](double, double, double){return 0.0;};

    // Initial condition
    auto u0 = [](double x, double y) {return std::sin(M_PI * x) * std::sin(M_PI * y) + 0.5 * std::sin(2 * M_PI * x) * std::sin(M_PI * y);};

    // Exact solution
    auto exact = [alpha](double x, double y, double t)
    {
        double mode1 = std::exp(-2 * M_PI * M_PI * alpha * t) * std::sin(M_PI * x) * std::sin(M_PI * y);
        double mode2 = 0.5 * std::exp(-5 * M_PI * M_PI * alpha * t) * std::sin(2 * M_PI * x) * std::sin(M_PI * y);
        return mode1 + mode2;
    };

    // Discretise PDE
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);

    // Time integrator
    constexpr double dt = 0.001 / alpha;
    temporal::CrankNicolson ti(dt);

    // Create solver object
    HeatPDE2D solver(fd, ti, 0.0, u0);

    // Solve and get residual error
    constexpr double t_end = 0.1 / alpha;
    double res_err = solve_and_get_error(solver, mesh, exact, t_end);

    EXPECT_LT(res_err, 1e-3);
}

// =============================================================================
// Test 5 - Verify that source terms are treated correctly.
//
// For u(x,y,t) = exp(-t) * sin(πx) * sin(πy), we have ∂u/∂t = Δu + f, with 
// f(x,y,t) = (-1 + 2π²) * u(x,y,t).  Imposing Dirichlet BCs at all sides leads 
// to u_left = u_right = u_bottom = u_top = 0, and the initial condition u0 = 
// = u(x,y,0) = sin(πx) * sin(πy) is used.
//
// The solution is approximated from t = 0 to t = 0.1. 
//
// With the choice of parameters, O(error) ≈ O(dx²) + O(dt²) ≈ 5e-4 < 1e-3.
// =============================================================================
TEST(HeatPDE2D, CrankNicolsonWithSource)
{
    constexpr int n = 101;
    const spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Boundary conditions
    auto dirichletBC = [](double, double, double){return 0;};
    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_shared<spatial::DirichletBoundaryCondition>(dirichletBC);
    bc[spatial::DomainSide::Right] = std::make_shared<spatial::DirichletBoundaryCondition>(dirichletBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(dirichletBC);
    bc[spatial::DomainSide::Top] = std::make_shared<spatial::DirichletBoundaryCondition>(dirichletBC);

    // Exact solution
    auto exact = [](double x, double y, double t) {return std::exp(-t) * std::sin(M_PI * x) * std::sin(M_PI * y);};

    // Source term
    auto source = [exact](double x, double y, double t){return (-1.0 + 2.0 * M_PI * M_PI) * exact(x, y, t);};

    // Initial condition
    auto u0 = [exact](double x, double y){return exact(x, y, 0.0);};

    // Discretize PDE
    constexpr double alpha = 1.0;
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);
    
    // Time integrator
    constexpr double dt = 0.01;
    temporal::CrankNicolson ti(dt);

    // Create solver object
    HeatPDE2D solver(fd, ti, 0.0, u0);

    // Solve and get residual error
    double t_end = 0.1;
    double res_err = solve_and_get_error(solver, mesh, exact, t_end);

    EXPECT_LT(res_err, 1e-3);
}

// =============================================================================
// Test 6 - Verify that the solver can integrate in different stages.
//
// Testing whether the solution is the same if integrated from t = t_start to 
// t = t_end (one stage), or integrated in two stages, from t_start to 
// (t_end + t_start) * 0.5 and from t_end / 2 to t_end.
// =============================================================================
TEST_F(DirichletBCTimeConvergence, IntegrateInStages)
{
    constexpr int n = 51;
    const spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Discretise PDE
    spatial::FiniteDifference2D fd_staged(alpha, mesh, bc, source);
    spatial::FiniteDifference2D fd_direct(alpha, mesh, bc, source);

    // Time integrators
    double dt = 0.01 / alpha;
    temporal::CrankNicolson ti_staged(dt);
    temporal::CrankNicolson ti_direct(dt);

    // Create solver object
    HeatPDE2D solver_staged(fd_staged, ti_staged, t_start, u0);
    HeatPDE2D solver_direct(fd_direct, ti_direct, t_start, u0);

    // Integrate
    solver_staged.integrate((t_end + t_start) * 0.5);
    solver_staged.integrate(t_end);

    solver_direct.integrate(t_end);

    EXPECT_NEAR((solver_staged.getSolution() - solver_direct.getSolution()).lpNorm<Eigen::Infinity>(), 0.0, 1e-12);
}

// =============================================================================
// Test 7 - Verify that the solver throws if t_end <= t_current.
// =============================================================================
TEST_F(DirichletBCTimeConvergence, InvalidTendThrows)
{
    constexpr int n = 51;
    const spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Discretise PDE
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);

    // Time integrators
    constexpr double dt = 0.01 / alpha;
    temporal::CrankNicolson ti(dt);

    // Create solver object
    HeatPDE2D solver(fd, ti, t_start, u0);

    EXPECT_THROW(solver.integrate(0.0), std::invalid_argument);
    EXPECT_THROW(solver.integrate(-1.0), std::invalid_argument);
}