#include "Mesh2D.hpp"

namespace spatial
{

Mesh2D::Mesh2D(double left, double right, double bottom, double top) 
{
    if (left >= right || bottom >= top) throw std::invalid_argument("Inconsistent geometrical constraints. The left (or bottom) side cannot be equal or larger than the right (or top) side.");

    domain_ = {left, right, bottom, top};
};

Mesh2D::Mesh2D(const Domain2D& domain) 
: Mesh2D(domain.left_, domain.right_, domain.bottom_, domain.top_) {};

// First is outward direction, second is inward direction.
const std::pair<DomainSide, DomainSide> Mesh2D::getBoundaryNormalDirections(DomainSide side) const
{
    if (side == DomainSide::Left) return {DomainSide::Left, DomainSide::Right};
    if (side == DomainSide::Right) return {DomainSide::Right, DomainSide::Left};
    if (side == DomainSide::Bottom) return {DomainSide::Bottom, DomainSide::Top};
    return {DomainSide::Top, DomainSide::Bottom};
};

/*
Clock-wise direction using the side's outward normal: first is left direction, second is right direction. That is:
    Left -> [Bottom, Top]
    Right -> [Top, Bottom]
    Bottom -> [Left, Right]
    Top -> [Right, Left]
*/
const std::pair<DomainSide, DomainSide> Mesh2D::getBoundaryTangentialDirections(DomainSide side) const
{
    if (side == DomainSide::Left) return getBoundaryNormalDirections(DomainSide::Bottom);
    if (side == DomainSide::Right) return getBoundaryNormalDirections(DomainSide::Top);
    if (side == DomainSide::Bottom) return getBoundaryNormalDirections(DomainSide::Left);
    return getBoundaryNormalDirections(DomainSide::Right);
};

}; // namespace