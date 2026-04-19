#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "CrankNicolson.hpp"
#include "DirichletBoundaryCondition.hpp"
#include "FiniteDifference2D.hpp"
#include "HeatPDE2D.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "SolutionWriter.hpp"
#include "StructuredMesh2D.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Helper: run a simulation and write output every `write_every` steps
// =============================================================================
void run(HeatPDE2D& solver,
         const spatial::StructuredMesh2D& mesh,
         SolutionWriter& writer,
         double t_end,
         int    write_every = 1)
{
    int step = 0;
    solver.integrate(t_end, [&](double t, const Eigen::VectorXd& u)
    {
        if (step % write_every == 0) writer.write(mesh, u, t);
        ++step;
    });
}

// =============================================================================
// Thermal mirage
//
// This is a naturally occurring optical phenomenon where light rays bend due to 
// spatial variations in the refractive index of the air, which are often caused 
// by temperature gradients. In this example, a simplified version is simulated 
// by imposing a time-varying temperature distribution at the bottom boundary 
// and allowing the heat to diffuse through the domain.
// =============================================================================
void example_thermal_mirage(const std::string& output_filename)
{
    constexpr int    n     = 101;
    constexpr double dt    = 0.01;
    constexpr double t_end = 5;

    spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Boundary conditions
    auto zeroBC = [](double, double, double){return 0.0;};
    auto bottomBC = [](double x, double, double t) 
    {
        return 0.5 + 0.1 * std::sin(2 * M_PI * x + t) + 0.2 * std::sin(6 * M_PI * x - 2 * t);
    };
    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left]   = std::make_shared<spatial::NeumannBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right]  = std::make_shared<spatial::NeumannBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(bottomBC);
    bc[spatial::DomainSide::Top]    = std::make_shared<spatial::NeumannBoundaryCondition>(zeroBC);

    // Thermal diffusivity
    auto alpha = [](double, double y) 
    {
        return 0.02 + 0.01 * std::exp(-y);
    };

    // Source term
    auto source = [](double, double, double){return 0.0;};
    
    // Initial condition
    auto u0 = [](double x, double y) 
    {
        return 0;
    };

    // Set up the solver and writer
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);
    temporal::CrankNicolson     ti(dt);
    HeatPDE2D                   solver(fd, ti, 0.0, u0);
    SolutionWriter              writer(output_filename);

    // Run and write output every 10 steps (every 0.1 time units)
    run(solver, mesh, writer, t_end, 10);
}

// =============================================================================
int main()
{
    std::cout << "Running: Thermal mirage...\n";

    std::string output_filename = "examples/thermal-mirage.csv";
    example_thermal_mirage(output_filename);

    std::cout << "  -> " << output_filename << " generated.\n";
    return 0;
}
