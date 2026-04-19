#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include <Eigen/Dense>

#include "CrankNicolson.hpp"
#include "FiniteDifference2D.hpp"
#include "HeatPDE2D.hpp"
#include "DirichletBoundaryCondition.hpp"
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
// Four colliding Gaussian pulses
//
// Four symmetric heat pulses drift toward each other, merge, and slowly
// spread out under zero Dirichlet walls. 
// =============================================================================
void example_colliding_pulses()
{
    constexpr int    n     = 101;
    constexpr double dt    = 0.05;
    constexpr double t_end = 5.0;

    spatial::StructuredMesh2D mesh(0, 1, 0, 1, n, n);

    // Boundary conditions
    auto zeroBC = [](double, double, double){ return 0.0; };
    spatial::BoundaryConditions bc;
    bc[spatial::DomainSide::Left]   = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Right]  = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Bottom] = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);
    bc[spatial::DomainSide::Top]    = std::make_shared<spatial::DirichletBoundaryCondition>(zeroBC);

    // Thermal diffusivity
    auto alpha = [](double, double y){return 0.01 * std::exp(-25.0 * (y - 0.5)*(y - 0.5));};

    // Source term
    auto source = [](double, double, double){return 0.0;};

    // Initial condition: four Gaussian pulses
    auto u0 = [](double x, double y) 
    {
        double p1 = std::exp(-80.0 * ((x - 0.25)*(x - 0.25) + (y - 0.25)*(y - 0.25)));
        double p2 = std::exp(-80.0 * ((x - 0.25)*(x - 0.25) + (y - 0.75)*(y - 0.75)));
        double p3 = std::exp(-80.0 * ((x - 0.75)*(x - 0.75) + (y - 0.25)*(y - 0.25)));
        double p4 = std::exp(-80.0 * ((x - 0.75)*(x - 0.75) + (y - 0.75)*(y - 0.75)));
        return p1 + p2 + p3 + p4;
    };

    // Set up the solver and writer
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);
    temporal::CrankNicolson     ti(dt);
    HeatPDE2D                   solver(fd, ti, 0.0, u0);
    SolutionWriter              writer("examples/colliding-pulses.csv");

    // Run and write output every 2 steps (every 0.1 time units)
    run(solver, mesh, writer, t_end, 2);
}

// =============================================================================
int main()
{
    std::cout << "Running: Four colliding Gaussian pulses...\n";

    example_colliding_pulses();

    std::cout << "  -> colliding-pulses.csv generated.\n";
    return 0;
}