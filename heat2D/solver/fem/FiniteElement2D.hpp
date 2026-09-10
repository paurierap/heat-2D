#ifndef FINITE_ELEMENT_2D_HPP
#define FINITE_ELEMENT_2D_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

#include "BoundaryConditions.hpp"
#include "Quadratures.hpp"
#include "SpatialDiscretization2D.hpp"

namespace heat2d::solver {

// Galerkin finite element discretization of the heat equation
class FiniteElement2D : public SpatialDiscretization2D {
 private:
  int quadratureOrder_;

  // Quadrature rules over the reference triangle and a reference edge
  quadrature::TriangleQuadrature interior_rule_;
  quadrature::GaussLegendre edge_rule_;

  // Return the number of free nodes of an element and populate the local array with the local indices.
  std::size_t classifyElement(const std::vector<std::size_t>& nodes,
                              std::array<std::size_t, 3>& local) const;

  // Assemble Robin boundary stiffness contributions into matrixK_.
  void applyBoundaryStiffness();

  // For eliminated Dirichlet DOFs, add the stiffness and mass couplings
  // -K_fd * u_d - M_fd * dg/dt to b_.
  void applyDirichletCouplingRHS(double t);

 public:
  FiniteElement2D(std::function<double(double, double)> alpha,
                  const mesh::Mesh2D& mesh,
                  BoundaryConditions boundary_conditions,
                  std::function<double(double, double, double)> source,
                  int quadratureOrder = 2);

  void discretize() override;
  void updateRHS(double t = 0.0) override;
  bool isMSPD() const override { return true; }
  bool isKSPD() const override { return true; }
};

}  // namespace heat2d::solver

#endif  // FINITE_ELEMENT_2D_HPP
