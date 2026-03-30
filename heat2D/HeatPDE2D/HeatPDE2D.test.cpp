#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

#include "HeatPDE2D.hpp"
#include "DirichletBoundaryCondition.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "FiniteDifference2D.hpp"
#include "StructuredMesh2D.hpp"
#include "ExplicitEuler.hpp"
#include "CrankNicolson.hpp"

double solve_and_get_error(HeatPDE2D& solver, const spatial::Mesh2D& mesh, std::function<double (double, double, double)> solution, double t_end)
{
    solver.integrate(t_end);

    Eigen::VectorXd sol = solver.getSolution();
    Eigen::VectorXd exact(sol.size());

    int j = 0;
    for (const auto& node : mesh.getNodes()) exact[j++] = solution(node.x_, node.y_, t_end);

    Eigen::VectorXd residual = exact - sol;
    return residual.lpNorm<Eigen::Infinity>();
};

Eigen::VectorXd solve_and_get_solution(HeatPDE2D& solver, double t_end)
{
    solver.integrate(t_end);
    return solver.getSolution();
}

double solve_and_get_error_vs_ref(HeatPDE2D& solver, const Eigen::Ref<const Eigen::VectorXd>& ref, double t_end)
{
    solver.integrate(t_end);
    Eigen::VectorXd sol = solver.getSolution();
    return (sol - ref).lpNorm<Eigen::Infinity>();
}

TEST(HeatPDE2D, TimeConvergence_Eigenmode_Dirichlet)
{
    int n = 151;
    double alpha = 0.5 / (M_PI * M_PI);
    spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Define BCs
    auto leftBC = [](double, double, double){return 0.0;};
    auto rightBC = [](double, double, double){return 0.0;};
    auto bottomBC = [](double, double, double){return 0.0;};
    auto topBC = [](double, double, double){return 0.0;};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_unique<spatial::DirichletBoundaryCondition>(leftBC);
    bc[spatial::DomainSide::Right] = std::make_unique<spatial::DirichletBoundaryCondition>(rightBC);
    bc[spatial::DomainSide::Bottom] = std::make_unique<spatial::DirichletBoundaryCondition>(bottomBC);
    bc[spatial::DomainSide::Top] = std::make_unique<spatial::DirichletBoundaryCondition>(topBC);

    // Source term:
    auto source = [&](double x, double y, double){return 0.0;};

    // Discretise PDE (fresh instances to avoid aliasing issues)
    spatial::FiniteDifference2D EEfd_coarse(alpha, mesh, bc, source);
    spatial::FiniteDifference2D EEfd_fine(alpha, mesh, bc, source);
    spatial::FiniteDifference2D CNfd_coarse(alpha, mesh, bc, source);
    spatial::FiniteDifference2D CNfd_fine(alpha, mesh, bc, source);
    spatial::FiniteDifference2D CNfd_ref(alpha, mesh, bc, source);

    // Initial condition and time integrators
    double dt = 0.01;
    double dt_ref = 0.001;
    double t_start = 0.0;
    auto u0 = [](double x, double y){return std::sin(M_PI * x) * std::sin(M_PI * y);};
    temporal::ExplicitEuler EEti_coarse(dt), EEti_fine(0.5 * dt);
    temporal::CrankNicolson CNti_coarse(dt), CNti_fine(0.5 * dt);
    temporal::CrankNicolson CNti_ref(dt_ref);

    // Create solver object
    double t_end = 1;
    HeatPDE2D EEsolver_coarse(EEfd_coarse, EEti_coarse, t_start, u0);
    HeatPDE2D EEsolver_fine(EEfd_fine, EEti_fine, t_start, u0);
    HeatPDE2D CNsolver_coarse(CNfd_coarse, CNti_coarse, t_start, u0);
    HeatPDE2D CNsolver_fine(CNfd_fine, CNti_fine, t_start, u0);
    HeatPDE2D CNsolver_ref(CNfd_ref, CNti_ref, t_start, u0);

    // Compare approximations
    Eigen::VectorXd ref = solve_and_get_solution(CNsolver_ref, t_end);
    double EE_err_coarse = solve_and_get_error_vs_ref(EEsolver_coarse, ref, t_end);
    double EE_err_fine = solve_and_get_error_vs_ref(EEsolver_fine, ref, t_end);
    double CN_err_coarse = solve_and_get_error_vs_ref(CNsolver_coarse, ref, t_end);
    double CN_err_fine = solve_and_get_error_vs_ref(CNsolver_fine, ref, t_end);

    // Verify Explicit Euler expected convergence rate
    double EE_rate = std::log(EE_err_coarse / EE_err_fine) / std::log(2);
    EXPECT_NEAR(EE_rate, 1, 0.2);

    // Verify Crank Nicolson expected convergence rate
    double CN_rate = std::log(CN_err_coarse / CN_err_fine) / std::log(2);
    EXPECT_NEAR(CN_rate, 2, 0.2);
}

TEST(HeatPDE2D, TwoModeDecay_Dirichlet)
{
    int n = 41;
    double alpha = 0.5 / (M_PI * M_PI);
    spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    auto zeroBC = [](double, double, double){return 0.0;};
    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_unique<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right] = std::make_unique<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Bottom] = std::make_unique<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Top] = std::make_unique<spatial::DirichletBoundaryCondition>(zeroBC);

    auto source = [](double, double, double){return 0.0;};

    auto u0 = [](double x, double y)
    {
        return std::sin(M_PI * x) * std::sin(M_PI * y)
             + 0.5 * std::sin(2 * M_PI * x) * std::sin(M_PI * y);
    };

    auto exact = [alpha](double x, double y, double t)
    {
        double mode1 = std::exp(-2 * M_PI * M_PI * alpha * t) * std::sin(M_PI * x) * std::sin(M_PI * y);
        double mode2 = 0.5 * std::exp(-5 * M_PI * M_PI * alpha * t) * std::sin(2 * M_PI * x) * std::sin(M_PI * y);
        return mode1 + mode2;
    };

    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);
    temporal::CrankNicolson ti(0.01);
    HeatPDE2D solver(fd, ti, 0.0, u0);

    double t_end = 1;
    double rel_err = solve_and_get_error(solver, mesh, exact, t_end);

    EXPECT_LT(rel_err, 2e-3);
}

TEST(HeatPDE2D, ManufacturedSolution_WithSource_Dirichlet)
{
    int n = 41;
    double alpha = 0.8 / (M_PI * M_PI);
    spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    auto exact = [](double x, double y, double t)
    {
        return std::exp(-t) * std::sin(M_PI * x) * std::sin(M_PI * y);
    };

    auto dirichletBC = [exact](double x, double y, double t){return exact(x, y, t);};

    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left] = std::make_unique<spatial::DirichletBoundaryCondition>(dirichletBC);
    bc[spatial::DomainSide::Right] = std::make_unique<spatial::DirichletBoundaryCondition>(dirichletBC);
    bc[spatial::DomainSide::Bottom] = std::make_unique<spatial::DirichletBoundaryCondition>(dirichletBC);
    bc[spatial::DomainSide::Top] = std::make_unique<spatial::DirichletBoundaryCondition>(dirichletBC);

    auto source = [alpha, exact](double x, double y, double t)
    {
        return (-1.0 + 2.0 * alpha * M_PI * M_PI) * exact(x, y, t);
    };

    auto u0 = [exact](double x, double y){return exact(x, y, 0.0);};

    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);
    temporal::CrankNicolson ti(0.01);
    HeatPDE2D solver(fd, ti, 0.0, u0);

    double t_end = 1;
    double rel_err = solve_and_get_error(solver, mesh, exact, t_end);

    EXPECT_LT(rel_err, 2e-3);
}
