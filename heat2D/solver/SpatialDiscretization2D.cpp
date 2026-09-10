#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <string>

#include "BoundaryConditions.hpp"
#include "Mesh2D.hpp"
#include "SpatialDiscretization2D.hpp"

namespace heat2d::solver {

SpatialDiscretization2D::SpatialDiscretization2D(std::function<double(double, double)> alpha,
                          const mesh::Mesh2D& mesh,
                          BoundaryConditions boundary_conditions,
                          std::function<double(double, double, double)> source)
      : alpha_(alpha),
        mesh_(mesh),
        source_(source),
        global_to_local_(mesh_.getNodes().size(), invalid_node_index),
        is_dirichlet_(mesh_.getNodes().size(), false),
        boundary_conditions_(boundary_conditions) {
  
  // Precompute Dirichlet nodes to be eliminated from the reduced system.
  for (const auto& [tag, BC] : boundary_conditions_) {
    if (BC->getType() == bc::BoundaryConditionType::Dirichlet)
      for (std::size_t nodeID : mesh_.getBoundary(tag))
        is_dirichlet_[nodeID] = true;
  }

  buildMappings();

  std::size_t local_space_size = local_to_global_.size();
  matrixK_.resize(local_space_size, local_space_size);
  matrixM_.resize(local_space_size, local_space_size);
  b_.resize(local_space_size);
};

// Factorize and solve matrixK_ * u = b_ in the reduced space.
Eigen::VectorXd SpatialDiscretization2D::solve_reduced() {
  Eigen::VectorXd reducedSolution(local_to_global_.size());
  updateRHS();

  // Direct LDL^T factorization (only if matrixK_ is SPD)
  if (isKSPD()) {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    ldlt.compute(-matrixK_);
    if (ldlt.info() != Eigen::Success)
      throw std::runtime_error("LDLT factorization failed\n");

    reducedSolution = ldlt.solve(b_);

    Eigen::VectorXd residual = (-matrixK_) * reducedSolution - b_;
    const double b_norm = b_.norm();
    if (b_norm > 0.0 && residual.norm() / b_norm > 1e-10)
      throw std::runtime_error("LDLT solve residual too large");
  } else  // Fall back to LU
  {
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
    lu.compute(-matrixK_);
    if (lu.info() != Eigen::Success)
      throw std::runtime_error("LU factorization failed\n");

    reducedSolution = lu.solve(b_);

    Eigen::VectorXd residual = (-matrixK_) * reducedSolution - b_;
    const double b_norm = b_.norm();
    if (b_norm > 0.0 && residual.norm() / b_norm > 1e-10)
      throw std::runtime_error("LU solve residual too large");
  }

  return reducedSolution;
}

// Solve Poisson's equation, i.e. du/dt = 0.
Eigen::VectorXd SpatialDiscretization2D::solveSteadyState() {
  std::cout << "\nSolving steady-state problem...\n";
  Eigen::VectorXd reducedSolution = solve_reduced();
  std::cout << "  -> Steady-state solution was successful!\n";

  return fillDirichletNodes(reducedSolution, 0.0);
}

Eigen::VectorXd SpatialDiscretization2D::fillDirichletNodes(
    const Eigen::Ref<const Eigen::VectorXd>& reducedSolution, double t) const {
  Eigen::VectorXd solution(mesh_.getNodes().size());

  // Fill solution with Dirichlet nodes
  const std::vector<mesh::Node2D>& nodes = mesh_.getNodes();
  for (const auto& node : nodes) {
    std::size_t globalID = node.nodeID_;

    if (!is_dirichlet_[globalID])
      solution[globalID] = reducedSolution[global_to_local_[globalID]];
  }

  for (const auto& [tag, BC] : boundary_conditions_) {
    if (BC->getType() == bc::BoundaryConditionType::Dirichlet) {
      for (std::size_t globalID : mesh_.getBoundary(tag)) {
        mesh::BoundaryNode2D boundary_node = mesh_.getBoundaryNode(globalID);
        double x = boundary_node.x_;
        double y = boundary_node.y_;

        solution[globalID] = BC->f(x, y, t);
      }
    }
  }

  return solution;
}

Eigen::VectorXd SpatialDiscretization2D::reduce(
    std::function<double(double, double)> u) {
  std::size_t reducedSize = local_to_global_.size();
  Eigen::VectorXd reducedU(reducedSize);

  for (std::size_t i = 0; i < reducedSize; ++i) {
    std::size_t globalID = local_to_global_[i];
    const mesh::Node2D& node = mesh_.getNode(globalID);

    reducedU[i] = u(node.x_, node.y_);
  }

  return reducedU;
}

}  // namespace heat2d::solver