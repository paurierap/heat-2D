#include "FiniteDifference2D.hpp"

#include <Eigen/Sparse>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include "BoundaryConditions.hpp"
#include "StructuredMesh2D.hpp"

namespace heat2d::solver {

FiniteDifference2D::FiniteDifference2D(
    std::function<double(double, double)> alpha,
    const mesh::StructuredMesh2D& mesh, BoundaryConditions boundary_conditions,
    std::function<double(double, double, double)> source)
    : SpatialDiscretization2D(alpha, mesh, boundary_conditions, source),
      mesh_(mesh) {

  // Resize arrays for reduced space
  std::size_t local_space_size = local_to_global_.size();
  tripletListK_.reserve(5 * local_space_size);
  matrixM_.setIdentity();
}

void FiniteDifference2D::discretize() {
  std::cout
      << "\nDiscretizing the spatial domain using finite differences...\n";

  applyLaplacian();
  applyBoundaryConditions();
  matrixK_.setFromTriplets(tripletListK_.begin(), tripletListK_.end());

  std::cout << "  -> Spatial discretization was successful.\n";
}

// Diagonal contribution to u_{i,j}
void FiniteDifference2D::addDiagonalTerm(std::size_t nodeID) {
  std::size_t localID = global_to_local_[nodeID];
  double x = mesh_.getNode(nodeID).x_;
  double y = mesh_.getNode(nodeID).y_;
  double dx = mesh_.getDx();
  double dy = mesh_.getDy();

  tripletListK_.emplace_back(
      localID, localID,
      -(alpha_(x + 0.5 * dx, y) + alpha_(x - 0.5 * dx, y)) / (dx * dx) -
          (alpha_(x, y + 0.5 * dy) + alpha_(x, y - 0.5 * dy)) / (dy * dy));
}

// Off diagonal contributions (multiplier parameter, defaulted to 1.0, included
// in case there is a contribution from Neumann/Robin BCs)
void FiniteDifference2D::addOffDiagonalTerm(
    std::size_t nodeID, const std::pair<int, int>& direction,
    double multiplier) {
  auto [dirx, diry] = direction;
  std::optional<std::size_t> neighbor = mesh_.getNeighbor(nodeID, direction);

  // Check if neighbor exists (in case of boundary nodes)
  if (!neighbor) return;

  // Check if neighbor has prescribed Dirichlet BCs
  if (is_dirichlet_[*neighbor]) return;

  std::size_t localID = global_to_local_[nodeID];
  std::size_t neighbor_local = global_to_local_[*neighbor];

  // Get coordinates of the node to evaluate alpha at the midpoint of the
  // stencil
  double x = mesh_.getNode(nodeID).x_;
  double y = mesh_.getNode(nodeID).y_;

  // Horizontal nodes of the stencil
  if (dirx) {
    double dx = mesh_.getDx();
    tripletListK_.emplace_back(
        localID, neighbor_local,
        alpha_(x + 0.5 * dirx * dx, y) / (dx * dx) * multiplier);
    return;
  }

  // Vertical nodes of the stencil
  double dy = mesh_.getDy();
  tripletListK_.emplace_back(
      localID, neighbor_local,
      alpha_(x, y + 0.5 * diry * dy) / (dy * dy) * multiplier);
}

// Second order discretization approximation is applied to the inner nodes. If
// an inner node has a Dirichlet boundary node, this is later treated when
// applying boundary conditions.
void FiniteDifference2D::applyLaplacian() {
  for (std::size_t globalID : mesh_.getInnerNodes()) {
    // u_{i,j}
    addDiagonalTerm(globalID);

    // u_{i-1,j}, u_{i+1,j}, u_{i,j-1}, u_{i,j+1}
    for (const auto& direction : stencil)
      addOffDiagonalTerm(globalID, direction);
  }

  return;
}

// The contributions to the matrix A from the boundary conditions (mainly
// Neumann/Robin BC's) are here considered. Dirichlet BC's and the extra term in
// Neumann/Robin are treated separately in a vector b. This way, A is constant
// and computed only once at the beginning of execution.
void FiniteDifference2D::applyBoundaryConditions() {
  // A boundary node can have 1 or 2 (corners) sides. If it belongs to a side
  // with a Dirichlet BC, the node (and its row in A) is omitted. If it's a
  // corner, a Dirichlet BC has preference over Neumann. If Neumann-Neumann, BCs
  // are treated naturally.
  const std::vector<mesh::BoundaryNode2D>& boundary_nodes =
      mesh_.getBoundaryNodes();

  for (const auto& boundary_node : boundary_nodes) {
    if (is_dirichlet_[boundary_node.nodeID_]) continue;
    isMatrixSPD = false;  // Neumann or Robin BCs present
    applyFluxBoundaryCondition(boundary_node);
  }

  return;
}

// Use ghost nodes, whereby the boundary node is treated almost like an inner
// node with a 4-point stencil (see
// https://www.12000.org/my_notes/neumman_BC/Neumman_BC.htm).
void FiniteDifference2D::applyFluxBoundaryCondition(
    const mesh::BoundaryNode2D& boundary_node) {
  std::size_t globalID = boundary_node.nodeID_;

  addDiagonalTerm(globalID);

  for (auto [dirx, diry] : stencil) {
    bool isInward = false;
    const std::string* activeTag = nullptr;
    for (const auto& tag : boundary_node.tags_) {
      auto [inx, iny] = mesh_.getBoundaryInwardDirection(tag);
      if (inx == dirx && iny == diry) {
        isInward = true;
        activeTag = &tag;
        break;
      }
    }

    addOffDiagonalTerm(globalID, {dirx, diry}, isInward ? 2.0 : 1.0);

    if (isInward) {
      std::size_t localID = global_to_local_[globalID];
      double x = boundary_node.x_;
      double y = boundary_node.y_;

      const bc::BoundaryCondition& BC = getBoundaryCondition(*activeTag);
      double h = dirx ? mesh_.getDx() : mesh_.getDy();
      double alpha_mid =
          dirx ? alpha_(x + 0.5 * dirx * h, y) : alpha_(x, y + 0.5 * diry * h);

      // Correction for the boundary node
      tripletListK_.emplace_back(
          localID, localID,
          -2.0 * alpha_mid / h * BC.u_coeff(x, y) / BC.du_coeff(x, y));
    }
  }
}

void FiniteDifference2D::updateRHS(double t) {
  b_.setZero();

  // A boundary node can have 1 or 2 (corners) sides. If it belongs to a side
  // with a Dirichlet BC, the node (and its row in A) is omitted. If it's a
  // corner, a Dirichlet BC has preference over Neumann. If Neumann-Neumann, BCs
  // are treated naturally.
  for (const auto& boundary_node : mesh_.getBoundaryNodes()) {
    if (is_dirichlet_[boundary_node.nodeID_])
      updateDirichletBoundaryCondition(boundary_node, t);
    else
      updateFluxBoundaryCondition(boundary_node, t);
  }

  // Source term
  const auto& nodes = mesh_.getNodes();
  for (std::size_t globalID : local_to_global_) {
    std::size_t localID = global_to_local_[globalID];
    b_[localID] += source_(nodes[globalID].x_, nodes[globalID].y_, t);
  }

  return;
}

void FiniteDifference2D::updateDirichletBoundaryCondition(
    const mesh::BoundaryNode2D& boundary_node, double t) {
  std::size_t globalID = boundary_node.nodeID_;
  double x = boundary_node.x_;
  double y = boundary_node.y_;

  for (const auto& tag : boundary_node.tags_) {
    // Get directions and values for the stencil
    auto [inx, iny] = mesh_.getBoundaryInwardDirection(tag);
    std::size_t neighbor_inward = *mesh_.getNeighbor(globalID, {inx, iny});

    // Check only for corner nodes with Dirichlet-Dirichlet BCs
    if (is_dirichlet_[neighbor_inward]) continue;

    std::size_t neighbor_local = global_to_local_[neighbor_inward];

    // Add contribution to the inward neighbor's row in vector b.

    if (inx) {
      double h = mesh_.getDx();
      b_[neighbor_local] += alpha_(x + 0.5 * inx * h, y) / (h * h) *
                            getBoundaryCondition(tag).f(x, y, t);
    } else {
      double h = mesh_.getDy();
      b_[neighbor_local] += alpha_(x, y + 0.5 * iny * h) / (h * h) *
                            getBoundaryCondition(tag).f(x, y, t);
    }
  }

  return;
}

void FiniteDifference2D::updateFluxBoundaryCondition(
    const mesh::BoundaryNode2D& boundary_node, double t) {
  std::size_t globalID = boundary_node.nodeID_;
  std::size_t localID = global_to_local_[globalID];
  double x = boundary_node.x_;
  double y = boundary_node.y_;

  for (const auto& tag : boundary_node.tags_) {
    auto [inx, iny] = mesh_.getBoundaryInwardDirection(tag);
    double h = inx ? mesh_.getDx() : mesh_.getDy();
    double alpha_mid =
        inx ? alpha_(x + 0.5 * inx * h, y) : alpha_(x, y + 0.5 * iny * h);

    const bc::BoundaryCondition& BC = getBoundaryCondition(tag);
    b_[localID] += 2.0 * alpha_mid / h * BC.f(x, y, t) / BC.du_coeff(x, y, t);
  }
}
};  // namespace heat2d::solver
