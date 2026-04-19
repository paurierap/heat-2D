#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "CrankNicolson.hpp"
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
// Insulated box with moving heat source
//
// All Neumann (perfectly insulated) walls. A Gaussian heat source orbits
// the centre of the domain, leaving a glowing trail as energy accumulates.
// =============================================================================
void example_moving_source(const std::string& output_filename)
{
    constexpr int    n     = 101;
    constexpr double dt    = 0.1;
    constexpr double t_end = 4.0 * M_PI; // Two full orbits

    spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Boundary conditions
    auto zero = [](double, double, double){ return 0.0; };
    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left]   = std::make_shared<spatial::NeumannBoundaryCondition>(zero);
    bc[spatial::DomainSide::Right]  = std::make_shared<spatial::NeumannBoundaryCondition>(zero);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::NeumannBoundaryCondition>(zero);
    bc[spatial::DomainSide::Top]    = std::make_shared<spatial::NeumannBoundaryCondition>(zero);

    // Thermal diffusivity
    auto alpha = [](double x, double y) 
    {
        double r = std::sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5));
        return 0.005 + 0.02 * std::exp(-8.0 * r * r);
    };

    // Source orbits at radius 0.25
    auto source = [](double x, double y, double t) 
    {
        double cx = 0.5 + 0.25 * std::cos(t);
        double cy = 0.5 + 0.25 * std::sin(t);
        return 1.0 * std::exp(-60.0 * ((x-cx)*(x-cx) + (y-cy)*(y-cy)));
    };

    // Initial condition
    auto u0 = [](double, double){return 0.0;};

    // Set up the solver and writer
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);
    temporal::CrankNicolson     ti(dt);
    HeatPDE2D                   solver(fd, ti, 0.0, u0);
    SolutionWriter              writer(output_filename);

    // Run and write output every 2 steps (every 0.2 time units)
    run(solver, mesh, writer, t_end, 2);
}

// =============================================================================
int main()
{
    std::cout << "Running: Moving heat source in insulated box...\n";

    std::string output_filename = "examples/moving-source.csv";
    example_moving_source(output_filename);

    std::cout << "  -> " << output_filename << " generated.\n";

    return EXIT_SUCCESS;
}