#ifndef SPATIALDISCRETIZATION2D_HPP
#define SPATIALDISCRETIZATION2D_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "BoundaryConditions.hpp"
#include "Mesh2D.hpp"

namespace heat2d::solver {

using BoundaryConditions = bc::BoundaryConditions;
using SparseMatrixRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

// Discretize the heat equation in space to build M * du/dt = K * u + b.
class SpatialDiscretization2D {
 protected:
  // TODO: Study change from reference to mesh to using a shared_ptr or even
  // removing mesh altogether.
  const mesh::Mesh2D& mesh_;

  static constexpr std::size_t invalid_node_index =
      std::numeric_limits<std::size_t>::max();

  std::function<double(double, double, double)> source_;
  std::function<double(double, double)> alpha_;

  // Sparse matrices for M and K and tripletlistK for assemblying K
  SparseMatrixRM matrixM_;
  SparseMatrixRM matrixK_;
  std::vector<Eigen::Triplet<double>> tripletListM_;
  std::vector<Eigen::Triplet<double>> tripletListK_;
  Eigen::VectorXd b_;

  // Mappings for nodes in the local, reduced space (Dirichlet nodes are
  // removed)
  std::vector<std::size_t> local_to_global_;
  std::vector<std::size_t> global_to_local_;

  // Check if node has prescribed Dirichlet BCs
  std::vector<bool> is_dirichlet_;

  // Store boundary conditions
  BoundaryConditions boundary_conditions_;

  // Factorize and solve matrixK_ * u = b_ in the reduced space.
  Eigen::VectorXd solve_reduced();

 public:
  explicit SpatialDiscretization2D(std::function<double(double, double)>,
                          const mesh::Mesh2D&,
                          BoundaryConditions,
                          std::function<double(double, double, double)>);

  virtual ~SpatialDiscretization2D() = default;

  // Map non-Dirichlet nodes to a reduced space, and Dirichlet nodes to invalid_node_index.
  inline void buildMappings() {
    const std::vector<mesh::Node2D>& nodes = mesh_.getNodes();

    std::size_t free_index = 0;
    for (const auto& node : nodes) {
      std::size_t globalID = node.nodeID_;
      if (is_dirichlet_[globalID]) continue;

      global_to_local_[globalID] = free_index;
      local_to_global_.push_back(globalID);
      free_index++;
    }
  }
  virtual void discretize() = 0;
  virtual void updateRHS(double t = 0.0) = 0;

  // Solves Au = b for steady-state problems. For time-dependent PDEs, this is
  // unused.
  Eigen::VectorXd solveSteadyState();

  // Reduce solution vector to only include non-Dirichlet nodes.
  Eigen::VectorXd reduce(std::function<double(double, double)>);
  Eigen::VectorXd fillDirichletNodes(const Eigen::Ref<const Eigen::VectorXd>&,
                                     double) const;

  // Getters
  inline const SparseMatrixRM& getMatrixM() const { return matrixM_; };
  inline const SparseMatrixRM& getMatrixK() const { return matrixK_; };
  inline const Eigen::VectorXd& getVector() const { return b_; };
  inline const bc::BoundaryCondition& getBoundaryCondition(
      const std::string& tag) const {
    return *boundary_conditions_.at(tag);
  }
  virtual bool isMSPD() const = 0;
  virtual bool isKSPD() const = 0;
};

};  // namespace heat2d::solver
#endif  // ifndef SPATIALDISCRETIZATION2D_HPP
