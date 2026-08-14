#ifndef FINITE_ELEMENT_2D_HPP
#define FINITE_ELEMENT_2D_HPP

#include <Eigen/Dense>
#include <array>
#include <functional>
#include <string>

#include "BoundaryCondition.hpp"
#include "SpatialDiscretization2D.hpp"
#include "StructuredMesh2D.hpp" 

namespace heat2d::solver {
    
class FiniteElement2D : public SpatialDiscretization2D {
 private: 
    
 public:
  FiniteElement2D(std::function<double(double, double)> alpha,
                          const mesh::Mesh2D& mesh,
                          BoundaryConditions boundary_conditions,
                          std::function<double(double, double, double)> source)
      : SpatialDiscretization2D(alpha, mesh, boundary_conditions, source) {};

};

}// namespace heat2d::solver

#endif // ifndef FINITE_ELEMENT_2D_HPP