#include "Mesh2D.hpp"

namespace spatial
{

Mesh2D::Mesh2D(double left, double right, double bottom, double top) 
{
    if (left > right || bottom > top) throw std::invalid_argument("Inconsistent geometrical constraints. The left (or bottom) side cannot be larger than the right (or top) side.");

    domain_ = {left, right, bottom, top};
};

Mesh2D::Mesh2D(const Domain2D& domain) 
: Mesh2D(domain.left_, domain.right_, domain.bottom_, domain.top_) {};

}; // namespace