#ifndef FINITEDIFFERENCE2D_HPP
#define FINITEDIFFERENCE2D_HPP

#include <Eigen/Dense>
#include <array>
#include <functional>
#include <string>

#include "BoundaryConditions.hpp"
#include "SpatialDiscretization2D.hpp"
#include "StructuredMesh2D.hpp"

namespace heat2d::solver {

class FiniteDifference2D : public SpatialDiscretization2D {
 private:
  static constexpr std::array<std::pair<int, int>, 4> stencil{
      {{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

  // Structured mesh required for finite differences
  const mesh::StructuredMesh2D& mesh_;

  // Flag to indicate if the Laplacian matrix K is symmetric positive definite
  // (SPD). If there are Neumann or Robin boundary conditions, the matrix K  may
  // not be SPD.
  bool isMatrixSPD = true;

  void addDiagonalTerm(std::size_t);
  void addOffDiagonalTerm(std::size_t, const std::pair<int, int>&,
                          double = 1.0);
  void applyLaplacian();
  void applyBoundaryConditions();
  void applyFluxBoundaryCondition(const mesh::BoundaryNode2D&);

  void updateDirichletBoundaryCondition(const mesh::BoundaryNode2D&, double t);
  void updateFluxBoundaryCondition(const mesh::BoundaryNode2D&, double t);
 public:
  FiniteDifference2D(std::function<double(double, double)>,
                     const mesh::StructuredMesh2D&, BoundaryConditions,
                     std::function<double(double, double, double)>);

  void discretize() override;
  void updateRHS(double t = 0.0) override;
  bool isMSPD() const override { return true; };
  bool isKSPD() const override { return isMatrixSPD; };
};

}  // namespace heat2d::solver
#endif  // ifndef FINITEDIFFERENCE2D_HPP