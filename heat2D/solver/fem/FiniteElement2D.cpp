#include "FiniteElement2D.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "IsoparametricMapping.hpp"
#include "Mesh2D.hpp"

namespace heat2d::solver {

FiniteElement2D::FiniteElement2D(
    std::function<double(double, double)> alpha, const mesh::Mesh2D& mesh,
    BoundaryConditions boundary_conditions,
    std::function<double(double, double, double)> source, int quadratureOrder)
    : SpatialDiscretization2D(alpha, mesh, boundary_conditions, source),
      quadratureOrder_(quadratureOrder) {
  // Cache the quadrature rules
  interior_rule_.compute(quadratureOrder_);
  edge_rule_.compute(quadratureOrder_);

  // Resize arrays for reduced space
  std::size_t local_space_size = local_to_global_.size();
  tripletListM_.reserve(9 * local_space_size);
  tripletListK_.reserve(9 * local_space_size);
}

std::size_t FiniteElement2D::classifyElement(
    const std::vector<std::size_t>& nodes,
    std::array<std::size_t, 3>& local) const {
  std::size_t n_free = 0;
  for (std::size_t k = 0; k < 3; ++k) {
    local[k] = global_to_local_[nodes[k]];
    n_free += (local[k] != invalid_node_index);
  }
  return n_free;
}

void FiniteElement2D::discretize() {
  std::cout << "\nDiscretizing the spatial domain using finite elements...\n";

  // Iterate over every element
  for (std::size_t e = 0; e < mesh_.getNumElements(); ++e) {
    std::vector<size_t> nodeIDs = mesh_.getElementNodes(e);
    IsoparametricMapping mapping(mesh_.getNodesFromIDs(nodeIDs));
    std::array<std::size_t, 3> local;

    // Skip elements with all-Dirichlet nodes.
    if (classifyElement(nodeIDs, local) == 0) continue;

    // Accumulate the element stiffness and mass matrices into dense local
    // matrices
    Eigen::Matrix3d Ke = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d Me = Eigen::Matrix3d::Zero();
    for (int q = 0; q < interior_rule_.numPoints(); ++q) {
      const double xi = interior_rule_.getPoints()(q, 0);
      const double eta = interior_rule_.getPoints()(q, 1);
      const double weight = 0.5 * interior_rule_.getWeights()(q) *
                            std::abs(mapping.detJacobian(xi, eta));

      const Eigen::Vector3d N = mapping.shapeFunctions(xi, eta);
      const Eigen::Matrix<double, 3, 2> gradN =
          mapping.physicalGradients(xi, eta);
      const mesh::Node2D phys = mapping.toPhysical(xi, eta);

      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          Ke(i, j) -= alpha_(phys.x_, phys.y_) * weight *
                      gradN.row(i).dot(gradN.row(j));
          Me(i, j) += weight * N(i) * N(j);
        }
      }
    }

    // Emit local entries to global sparse matrices (symmetric)
    for (int i = 0; i < 3; ++i) {
      if (local[i] == invalid_node_index) continue;
      for (int j = i; j < 3; ++j) {
        if (local[j] == invalid_node_index) continue;

        tripletListK_.emplace_back(local[i], local[j], Ke(i, j));
        tripletListM_.emplace_back(local[i], local[j], Me(i, j));

        if (i == j) continue;
        tripletListK_.emplace_back(local[j], local[i], Ke(i, j));
        tripletListM_.emplace_back(local[j], local[i], Me(i, j));
      }
    }
  }

  applyBoundaryStiffness();

  matrixK_.setFromTriplets(tripletListK_.begin(), tripletListK_.end());
  matrixM_.setFromTriplets(tripletListM_.begin(), tripletListM_.end());

  std::cout << "  -> Spatial discretization was successful.\n";
}

void FiniteElement2D::applyBoundaryStiffness() {
  for (const auto& [tag, BC] : boundary_conditions_) {
    if (BC->getType() != bc::BoundaryConditionType::Robin) continue;

    for (const auto& [a, b] : mesh_.getBoundaryEdgeNodes(tag)) {
      const mesh::Node2D& na = mesh_.getNode(a);
      const mesh::Node2D& nb = mesh_.getNode(b);
      double length = std::hypot(nb.x_ - na.x_, nb.y_ - na.y_);
      double scale = 0.5 * length;

      std::array<std::size_t, 2> local = {global_to_local_[a],
                                          global_to_local_[b]};

      // Accumulate the element stiffness matrices into a dense local matrix.
      Eigen::Matrix2d Ke = Eigen::Matrix2d::Zero();
      for (int q = 0; q < edge_rule_.numPoints(); ++q) {
        const double s = edge_rule_.getNodes()(q);
        const double x = 0.5 * (na.x_ * (1 - s) + nb.x_ * (1 + s));
        const double y = 0.5 * (na.y_ * (1 - s) + nb.y_ * (1 + s));
        const double u_c = BC->u_coeff(x, y, 0.0);
        if (u_c == 0.0) continue;  // Secretly Neumann.

        const double wq = edge_rule_.getWeights()(q) * scale;
        const double coef = -wq * alpha_(x, y) * u_c / BC->du_coeff(x, y, 0.0);

        const double Na = 0.5 * (1 - s);
        const double Nb = 0.5 * (1 + s);
        std::array<double, 2> Nvals = {Na, Nb};

        for (int i = 0; i < 2; ++i) {
          for (int j = i; j < 2; ++j) Ke(i, j) += coef * Nvals[i] * Nvals[j];
        }
      }

      // Emit local entries to global matrixK_ (symmetric)
      for (int i = 0; i < 2; ++i) {
        if (local[i] == invalid_node_index) continue;
        for (int j = i; j < 2; ++j) {
          if (local[j] == invalid_node_index) continue;
          tripletListK_.emplace_back(local[i], local[j], Ke(i, j));
          if (i == j) continue;
          tripletListK_.emplace_back(local[j], local[i], Ke(i, j));
        }
      }
    }
  }
}

void FiniteElement2D::updateRHS(double t) {
  b_.setZero();

  // Interior source term: b_i += int_Omega f * N_i dOmega.
  for (std::size_t e = 0; e < mesh_.getNumElements(); ++e) {
    std::vector<size_t> nodeIDs = mesh_.getElementNodes(e);
    IsoparametricMapping mapping(mesh_.getNodesFromIDs(nodeIDs));

    for (int q = 0; q < interior_rule_.numPoints(); ++q) {
      const double xi = interior_rule_.getPoints()(q, 0);
      const double eta = interior_rule_.getPoints()(q, 1);
      const double weight = 0.5 * interior_rule_.getWeights()(q) *
                            std::abs(mapping.detJacobian(xi, eta));
      const Eigen::Vector3d N = mapping.shapeFunctions(xi, eta);
      const mesh::Node2D phys = mapping.toPhysical(xi, eta);
      const double f = source_(phys.x_, phys.y_, t);

      for (int i = 0; i < 3; ++i) {
        std::size_t local = global_to_local_[nodeIDs[i]];
        if (local == invalid_node_index) continue;
        b_[local] += weight * f * N(i);
      }
    }
  }

  // Boundary flux terms for Neumann/Robin edges.
  for (const auto& [tag, BC] : boundary_conditions_) {
    if (BC->getType() == bc::BoundaryConditionType::Dirichlet) continue;

    for (const auto& [a, b] : mesh_.getBoundaryEdgeNodes(tag)) {
      const mesh::Node2D& na = mesh_.getNode(a);
      const mesh::Node2D& nb = mesh_.getNode(b);
      double length = std::hypot(nb.x_ - na.x_, nb.y_ - na.y_);
      double scale = 0.5 * length;

      std::size_t la = global_to_local_[a];
      std::size_t lb = global_to_local_[b];

      for (int q = 0; q < edge_rule_.numPoints(); ++q) {
        const double s = edge_rule_.getNodes()(q);
        const double x = 0.5 * (na.x_ * (1 - s) + nb.x_ * (1 + s));
        const double y = 0.5 * (na.y_ * (1 - s) + nb.y_ * (1 + s));

        const double flux = BC->f(x, y, t) / BC->du_coeff(x, y, t);
        const double wq =
            edge_rule_.getWeights()(q) * scale * alpha_(x, y) * flux;

        const double Na = 0.5 * (1 - s);
        const double Nb = 0.5 * (1 + s);

        if (la != invalid_node_index) b_[la] += wq * Na;
        if (lb != invalid_node_index) b_[lb] += wq * Nb;
      }
    }
  }

  // Coupling from eliminated Dirichlet DOFs.
  applyDirichletCouplingRHS(t);
}

void FiniteElement2D::applyDirichletCouplingRHS(double t) {
  // Precompute the Dirichlet values and their numeric time derivatives
  // (central differences) for all boundary nodes.
  const double eps = 1e-6 * std::max(1.0, std::abs(t));
  std::vector<double> uDirichlet(mesh_.getNodes().size(), 0.0);
  std::vector<double> gdot(mesh_.getNodes().size(), 0.0);
  for (const auto& [tag, BC] : boundary_conditions_) {
    if (BC->getType() != bc::BoundaryConditionType::Dirichlet) continue;
    for (std::size_t nodeID : mesh_.getBoundary(tag)) {
      const mesh::Node2D& n = mesh_.getNode(nodeID);
      uDirichlet[nodeID] = BC->f(n.x_, n.y_, t);
      gdot[nodeID] = (BC->f(n.x_, n.y_, t + eps) - BC->f(n.x_, n.y_, t - eps)) /
                     (2.0 * eps);
    }
  }

  for (std::size_t e = 0; e < mesh_.getNumElements(); ++e) {
    std::vector<size_t> nodeIDs = mesh_.getElementNodes(e);
    IsoparametricMapping mapping(mesh_.getNodesFromIDs(nodeIDs));
    std::array<std::size_t, 3> local;
    const std::size_t n_free = classifyElement(nodeIDs, local);

    // Only elements mixing free and Dirichlet nodes contribute to the
    // coupling terms.
    if (n_free == 0 || n_free == 3) continue;

    // Stiffness coupling pass: b_ -= K_fd * g.
    for (int q = 0; q < interior_rule_.numPoints(); ++q) {
      const double xi = interior_rule_.getPoints()(q, 0);
      const double eta = interior_rule_.getPoints()(q, 1);
      const double weight = 0.5 * interior_rule_.getWeights()(q) *
                            std::abs(mapping.detJacobian(xi, eta));
      const Eigen::Matrix<double, 3, 2> gradN =
          mapping.physicalGradients(xi, eta);
      const mesh::Node2D phys = mapping.toPhysical(xi, eta);
      const double a = alpha_(phys.x_, phys.y_);

      for (int i = 0; i < 3; ++i) {
        if (local[i] == invalid_node_index) continue;  // Dirichlet node
        for (int j = 0; j < 3; ++j) {
          if (local[j] != invalid_node_index) continue;  // free node
          b_[local[i]] -= weight * a * gradN.row(i).dot(gradN.row(j)) *
                          uDirichlet[nodeIDs[j]];
        }
      }
    }

    // Mass coupling pass: b_ -= M_fd * dg/dt.
    for (int q = 0; q < interior_rule_.numPoints(); ++q) {
      const double xi = interior_rule_.getPoints()(q, 0);
      const double eta = interior_rule_.getPoints()(q, 1);
      const double weight = 0.5 * interior_rule_.getWeights()(q) *
                            std::abs(mapping.detJacobian(xi, eta));
      const Eigen::Vector3d N = mapping.shapeFunctions(xi, eta);

      for (int i = 0; i < 3; ++i) {
        if (local[i] == invalid_node_index) continue;  // Dirichlet node
        for (int j = 0; j < 3; ++j) {
          if (local[j] != invalid_node_index) continue;  // free node
          b_[local[i]] -= weight * N(i) * N(j) * gdot[nodeIDs[j]];
        }
      }
    }
  }
}

}  // namespace heat2d::solver