#include <cmath>
#include <chrono>
#include <functional>
#include <iostream>
#include <omp.h>
#include <string>

#include "CrankNicolson.hpp"
#include "DirichletBoundaryCondition.hpp"
#include "FiniteDifference2D.hpp"
#include "HeatPDE2D.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "StructuredMesh2D.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// Benchmarking with a decaying Gaussian pulse
// =============================================================================
void benchmark()
{
    constexpr int    n        = 1001;
    constexpr int    n_steps  = 10;
    constexpr double dt       = 1e-4;
    double           t        = 0.0;

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

    // Initial condition
    auto u0 = [](double x, double y){return std::exp(-80.0 * ((x - 0.25)*(x - 0.25) + (y - 0.25)*(y - 0.25)));};

    // Set up the solver and writer
    spatial::FiniteDifference2D fd(alpha, mesh, bc, source);
    temporal::CrankNicolson     ti(dt);

    auto t0 = std::chrono::high_resolution_clock::now();
    fd.discretize();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    Eigen::VectorXd u = fd.reduce(u0);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    // Cache necessary matrices (depending on the time integration scheme)
    ti.setUp(fd);
    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "\nWith n = " << n << ": Discretize = " 
            << std::chrono::duration<double,std::milli>(t1-t0).count() << "ms,  "
            << "setUp = "
            << std::chrono::duration<double,std::milli>(t3-t2).count() << "ms\n\n";

    for (int i = 0; i < n_steps; ++i)
    {
        auto t4 = std::chrono::high_resolution_clock::now();
        ti.step(fd, t, u);
        auto t5 = std::chrono::high_resolution_clock::now();

        std::cout << "Step " << i << ": " << std::chrono::duration<double,std::milli>(t5-t4).count() << "ms\n";
        t += dt;
    }
}

// =============================================================================
int main()
{
    std::cout << "Benchmarking decaying Gaussian pulse...\n";

    //test_omp();

    benchmark();

    std::cout << "  -> Benchmark completed.\n";
    return 0;
}
