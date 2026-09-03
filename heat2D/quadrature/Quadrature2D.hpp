#ifndef QUADRATURE2D_HPP
#define QUADRATURE2D_HPP
#include <Eigen/Dense>

namespace heat2d::quadrature {

class Quadrature2D {
 protected:
  Eigen::MatrixX2d points_;
  Eigen::VectorXd  weights_;

 public:
  virtual ~Quadrature2D() = default;

  const Eigen::MatrixX2d& getPoints() const { return points_; }
  const Eigen::VectorXd& getWeights() const { return weights_; }
  int numPoints() const {return points_.rows(); };
};

} // namespace heat2d::quadrature

#endif  // ifndef QUADRATURE2D_HPP