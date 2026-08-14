#ifndef SPATIALDISCRETIZATION2D_HPP
#define SPATIALDISCRETIZATION2D_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "BoundaryCondition.hpp"
#include "Mesh2D.hpp"

namespace heat2d::solver {

// Pointer required for run-time polymorphism and to be used in different
// instances of the class
using BoundaryConditions =
    std::unordered_map<std::string, std::shared_ptr<bc::BoundaryCondition>>;
using SparseMatrixRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;

// Discretize the heat equation in space to build the ODE system M * du/dt = K * u + b. 
class SpatialDiscretization2D {
 private:
  // TODO: Study change from reference to mesh to using a shared_ptr or even
  // removing mesh altogether.
  const mesh::Mesh2D& mesh_;

 protected:
  std::function<double(double, double, double)> source_;
  std::function<double(double, double)> alpha_;

  // Sparse matrix and tripletlist for assembly
  SparseMatrixRM matrixM_;
  SparseMatrixRM matrixK_;
  std::vector<Eigen::Triplet<double>> tripletList;
  Eigen::VectorXd b_;

  // Mappings for nodes in the local, reduced space (Dirichlet nodes are
  // removed)
  std::vector<std::size_t> local_to_global_;
  std::vector<std::size_t> global_to_local_;

  // Check if node has prescribed Dirichlet BCs
  std::vector<bool> is_dirichlet_;

  // Store boundary conditions
  BoundaryConditions boundary_conditions_;

 public:
  SpatialDiscretization2D(std::function<double(double, double)> alpha,
                          const mesh::Mesh2D& mesh,
                          BoundaryConditions boundary_conditions,
                          std::function<double(double, double, double)> source)
      : alpha_(alpha),
        mesh_(mesh),
        source_(source),
        global_to_local_(mesh_.getNodes().size(), -1),
        is_dirichlet_(mesh_.getNodes().size(), false),
        boundary_conditions_(boundary_conditions) {};

  virtual ~SpatialDiscretization2D() = default;

  // Discretize and build matrix A and vector b
  virtual void buildMappings() = 0;
  virtual void discretize() = 0;
  virtual void applyLaplacian() = 0;
  virtual void applyBoundaryConditions() = 0;
  virtual void updateRHS(double t) = 0;

  // Solves Au = b for steady-state problems. For time-dependent PDEs, this is
  // unused.
  virtual Eigen::VectorXd solveSteadyState() = 0;
  virtual Eigen::VectorXd reduce(std::function<double(double, double)>) = 0;
  virtual Eigen::VectorXd fillDirichletNodes(
      const Eigen::Ref<const Eigen::VectorXd>&, double) const = 0;

  // Getters
  inline const SparseMatrixRM& getMatrixM() const { return matrixM_; };
  inline const SparseMatrixRM& getMatrixK() const { return matrixK_; };
  inline const Eigen::VectorXd& getVector() const { return b_; };
  inline const bc::BoundaryCondition& getBoundaryCondition(
      const std::string& tag) const {
    return *boundary_conditions_.at(tag);
  }
  virtual bool isSPD() const = 0;
};

};  // namespace heat2d::solver
#endif  // ifndef SPATIALDISCRETIZATION2D_HPP