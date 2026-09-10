#ifndef ISOPARAMETRIC_MAPPING_HPP
#define ISOPARAMETRIC_MAPPING_HPP

#include <array>
#include <Eigen/Dense>
#include <vector>

#include "Mesh2D.hpp"

namespace heat2d::solver {

// Maps the reference triangle with vertices (0,0), (1,0), (0,1) to a physical
// triangular element. 

// TODO: Generalize to higher-order elements and other element types (quads).
template <int NumNodes = 3>
class IsoparametricMapping {
 private:
 std::vector<mesh::Node2D> physicalNodes_;

 public:
  // Physical coordinates of the element's three nodes.
  explicit IsoparametricMapping(const std::vector<mesh::Node2D>& physicalCoords)
      : physicalNodes_(physicalCoords) {}

  // Shape function values at reference point (xi, eta).
  Eigen::Vector3d shapeFunctions(double xi, double eta) const {
    return Eigen::Vector3d{1.0 - xi - eta, xi, eta};
  }

  // Reference gradients of the shape functions (constant per element).
  Eigen::Matrix<double, NumNodes, 2> referenceGradients() const {
    Eigen::Matrix<double, NumNodes, 2> G;
    G << -1.0, -1.0,  //
        1.0, 0.0,     //
        0.0, 1.0;
    return G;
  }

  // Map reference point to physical coordinates.
  mesh::Node2D toPhysical(double xi, double eta) const {
    Eigen::Vector3d phi = shapeFunctions(xi, eta);

    return mesh::Node2D{0, physicalNodes_[0].x_ * phi[0] + physicalNodes_[1].x_ * phi[1] + physicalNodes_[2].x_ * phi[2],
                         physicalNodes_[0].y_ * phi[0] + physicalNodes_[1].y_ * phi[1] + physicalNodes_[2].y_ * phi[2]};
  }

  // Jacobian of the mapping: J = d(x,y) / d(xi,eta). For a linear triangle it is independent of the reference point. 
  // TODO: Generalize jacobian to higher-order elements.
  Eigen::Matrix2d jacobian(double xi, double eta) const {
    Eigen::Matrix2d J;
    J(0, 0) = physicalNodes_[1].x_ - physicalNodes_[0].x_;
    J(0, 1) = physicalNodes_[2].x_ - physicalNodes_[0].x_;
    J(1, 0) = physicalNodes_[1].y_ - physicalNodes_[0].y_;
    J(1, 1) = physicalNodes_[2].y_ - physicalNodes_[0].y_;
    return J;
  }

  // Determinant of the Jacobian.
  double detJacobian(double xi, double eta) const {
    return jacobian(xi, eta).determinant();
  }

  // Physical gradients of the shape functions.
  Eigen::Matrix<double, NumNodes, 2> physicalGradients(double xi, double eta) const {
    return referenceGradients() * jacobian(xi, eta).inverse();
  }
};

}  // namespace heat2d::solver

#endif  // ISOPARAMETRIC_MAPPING_HPP
