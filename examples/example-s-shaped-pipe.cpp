#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include "BoundaryConditions.hpp"
#include "CrankNicolson.hpp"
#include "HeatPDE2D.hpp"
#include "SolutionWriter.hpp"
#include "UnstructuredMesh2D.hpp"
#include "fem/FiniteElement2D.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace heat2d;

// =============================================================================
// Helper: run a simulation and write output every `write_every` steps
// =============================================================================
void run(HeatPDE2D& solver, const mesh::Mesh2D& mesh, SolutionWriter& writer,
         double t_end, int write_every = 1) {
  int step = 0;
  writer.write(mesh, solver.getSolution(), 0.0);  // initial condition
  solver.integrate(t_end, [&](double t, const Eigen::VectorXd& u) {
    if (step % write_every == write_every - 1) writer.write(mesh, u, t);
    ++step;
  });
}

// =============================================================================
// Insulated S-shaped pipe with alternating hot and cold reservoirs
//
// An unstructured finite element mesh (S-shaped pipe) is loaded from a Gmsh
// file. The pipe walls are perfectly insulated (null Neumann). Both end caps
// oscillate between zero and unit temperature with a half-period delay, so
// their hot and cold roles alternate as heat diffuses through the bends.
//
// Open examples/s-shaped-pipe/s-shaped-pipe.pvd in ParaView with a fixed [0,1]
// temperature colour range. Use a later cycle to reduce startup transients.
// =============================================================================
void example_s_shaped_pipe(const std::string& mesh_file,
                           const std::string& output_filename,
                           SolutionWriter::FORMAT format) {
  // Load the S-shaped pipe mesh (Gmsh MSH 4.1)
  mesh::UnstructuredMesh2D mesh(mesh_file);

  // Boundary conditions. The keys must match the mesh physical names.
  auto zeroBC = [](double, double, double) { return 0.0; };
  auto hotT = [](double, double, double t) {
    return t < 2.0 ? 0.5 * (1.0 - std::cos(M_PI * t / 2.0)) : 1.0;
  };
  // auto coldT = [=](double, double, double t) {
  //   return 0.5 - 0.5 * std::sin(2.0 * M_PI * t / period);
  // };

  bc::BoundaryConditions bc;
  bc["hot_end"] = std::make_shared<bc::DirichletBoundaryCondition>(hotT);
  bc["cold_end"] = std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);
  bc["insulated_top_hook"] =
      std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);
  bc["insulated_bottom_hook"] =
      std::make_shared<bc::NeumannBoundaryCondition>(zeroBC);

  // Thermal diffusivity
  auto alpha = [](double, double) { return 0.2; };

  // Source term
  auto source = [](double, double, double) { return 0.0; };

  // Both end caps start at 0.5, matching the uniform initial temperature.
  auto u0 = [](double, double) { return 0.0; };

  // Set up the solver and writer
  solver::FiniteElement2D fe(alpha, mesh, bc, source, 2);
  ode::CrankNicolson ti(0.05);
  HeatPDE2D solver(fe, ti, 0.0, u0);
  SolutionWriter writer(output_filename, format);

  // Four cycles, with snapshots every 0.5 time units (160 per cycle).
  run(solver, mesh, writer, 50, 20);
}

// =============================================================================
int main(int argc, char* argv[]) {
  const std::string mesh_file =
      argc > 1 ? argv[1] : "heat2D/mesh/test_shape.msh";
  const std::string output_filename = "examples/s-shaped-pipe";

  std::cout << "Running: Heat conduction through an insulated S-shaped "
               "pipe...\n";

  example_s_shaped_pipe(mesh_file, output_filename,
                        SolutionWriter::FORMAT::VTU);

  std::cout << "  -> " << output_filename << " generated.\n";
  return EXIT_SUCCESS;
}
