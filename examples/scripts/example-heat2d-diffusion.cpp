#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "BoundaryConditions.hpp"
#include "CrankNicolson.hpp"
#include "HeatPDE2D.hpp"
#include "SolutionWriter.hpp"
#include "UnstructuredMesh2D.hpp"
#include "fem/FiniteElement2D.hpp"

using namespace heat2d;

// =============================================================================
// Helper: run a simulation and write output every `write_every` steps
// =============================================================================
void run(HeatPDE2D& solver, const mesh::Mesh2D& mesh, SolutionWriter& writer,
         double t_end, int write_every = 1) {
  int step = 0;
  writer.write(mesh, solver.getSolution(), 0.0); 
  solver.integrate(t_end, [&](double t, const Eigen::VectorXd& u) {
    if (step % write_every == write_every - 1) writer.write(mesh, u, t);
    ++step;
  });
}

// Heat a connected HEAT2D wordmark from the bottom-left foot of H. The rest of the boundaries are insulated. The "sink" boundary is the inner hole of the letter D.

void example_heat2d_diffusion(const std::string& mesh_file,
                              const std::string& output_filename) {
  mesh::UnstructuredMesh2D mesh(mesh_file);

  auto zero = [](double, double, double) { return 0.0; };
  auto heater = [](double, double, double t) { return -std::expm1(-0.5 * t); };
  auto source = [](double, double, double) { return 0.0; };

  // All boundaries are insulated, except for the heater.
  bc::BoundaryConditions bc;
  bc["heater"] = std::make_shared<bc::DirichletBoundaryCondition>(heater);
  bc["sink"] = std::make_shared<bc::NeumannBoundaryCondition>(zero);
  bc["insulated"] = std::make_shared<bc::NeumannBoundaryCondition>(zero);

  auto alpha = [](double, double) { return 1.0; };
  auto u0 = [](double, double) { return 0.0; };
  solver::FiniteElement2D fe(alpha, mesh, bc, source, 2);
  ode::CrankNicolson ti(0.1);
  HeatPDE2D solver(fe, ti, 0.0, u0);
  SolutionWriter writer(output_filename, SolutionWriter::FORMAT::VTU);

  run(solver, mesh, writer, 50, 10); 
}

// Different options of running the example:
int main(int argc, char* argv[]) {
  const std::string mesh_file =
      argc > 1 ? argv[1] : "examples/heat2d_wordmark.msh";
  const std::string output_filename =
      argc > 2 ? argv[2] : "examples/heat2d-diffusion";

  std::cout << "Running: Heat diffusion through HEAT2D wordmark...\n";
  example_heat2d_diffusion(mesh_file, output_filename);
  std::cout << "  -> " << output_filename << " generated.\n";
  
  return EXIT_SUCCESS;
}
