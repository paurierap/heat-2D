#ifndef QUADRATURE2D_HPP
#define QUADRATURE2D_HPP

#include <array>
#include <vector>

namespace heat2d::quadrature 
{

class Quadrature2D 
{
private:
    int order_;
    std::vector<std::array<double,2>> points_;
    std::vector<double> weights_;
public:
    virtual ~Quadrature2D() = default;
    virtual std::size_t numPoints() const = 0;
    virtual const std::vector<std::array<double,2>>& points()  const = 0;
    virtual const std::vector<double>&               weights() const = 0;
    virtual int order() const = 0;
};

class TriangleGauss : public Quadrature2D {
public:
    explicit TriangleGauss(int order);
    std::size_t numPoints() const override { return weights_.size(); }
    const std::vector<std::array<double,2>>& points()  const override { return points_; }
    const std::vector<double>&               weights() const override { return weights_; }
    int order() const override { return order_; }

};

} // namespace

#endif