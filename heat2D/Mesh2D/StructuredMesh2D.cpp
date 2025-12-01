#include "StructuredMesh2D.hpp"
#include <optional>

namespace spatial
{

StructuredMesh2D::StructuredMesh2D(double left, double right, double bottom, double top, int nx, int ny) 
: Mesh2D(left, right, bottom, top)
{
    if (nx <= 1 || ny <= 1) throw std::invalid_argument("Number of nodes must be positive and greater than 1.");
    
    nx_ = nx;
    ny_ = ny;

    meshDomain();
};

StructuredMesh2D::StructuredMesh2D(const Domain2D& domain, int nx, int ny) 
: StructuredMesh2D(domain.left_, domain.right_, domain.bottom_, domain.top_, nx, ny) {};

void StructuredMesh2D::meshDomain()
{
    double dx = getDx(), dy = getDy();

    nodes_.reserve(nx_ * ny_);
    inner_nodes_.reserve((nx_ - 2) * (ny_ - 2));
    boundary_nodes_.reserve(2 * (nx_ + ny_) - 4);
    node_to_boundary_node.resize(nx_ * ny_, -1);

    boundaries_[DomainSide::Left].reserve(ny_);
    boundaries_[DomainSide::Right].reserve(ny_);
    boundaries_[DomainSide::Bottom].reserve(nx_);
    boundaries_[DomainSide::Top].reserve(nx_);

    int boundary_node = 0;
    for (int row = 0; row < ny_; ++row)
    {
        for (int col = 0; col < nx_; ++col)
        {
            int nodeID = nx_ * row + col;
            double x = domain_.left_ + col * dx;
            double y = domain_.bottom_ + row * dy;
            nodes_.push_back(Node2D{nodeID, x, y});

            std::vector<DomainSide> sides;

            if (col == 0) 
            {
                boundaries_[DomainSide::Left].push_back(nodeID);
                sides.push_back(DomainSide::Left);
            } 
            if (col == nx_ - 1) 
            {
                boundaries_[DomainSide::Right].push_back(nodeID);
                sides.push_back(DomainSide::Right);
            } 
            if (row == 0) 
            {
                boundaries_[DomainSide::Bottom].push_back(nodeID);
                sides.push_back(DomainSide::Bottom);
            } 
            if (row == ny_ - 1) 
            {
                boundaries_[DomainSide::Top].push_back(nodeID);
                sides.push_back(DomainSide::Top);
            }

            if (sides.empty()) inner_nodes_.push_back(nodeID);
            else
            {
                node_to_boundary_node[nodeID] = boundary_node;
                boundary_node++;
                boundary_nodes_.push_back(BoundaryNode2D{nodeID, x, y, sides});
            }
        }
    }
};


std::optional<int> StructuredMesh2D::getNodeID(int i, int j) const
{
    if (i < 0 || i >= nx_ || j < 0 || j >= ny_) return std::nullopt;
    return j * nx_ + i;
};

std::optional<int> StructuredMesh2D::getNeighbor(int nodeID, DomainSide side) const
{
    switch (side)
    {
        case DomainSide::Left:
            if (nodeID % nx_ == 0) return std::nullopt;
            return nodeID - 1;

        case DomainSide::Right:
            if ((nodeID + 1) % nx_ == 0) return std::nullopt;
            return nodeID + 1;

        case DomainSide::Bottom:
            if ((nodeID - nx_) < 0) return std::nullopt;
            return nodeID - nx_;

        case DomainSide::Top:
            if ((nodeID + nx_) > nx_ * ny_ - 1) return std::nullopt;
            return nodeID + nx_;
    }    
};

bool StructuredMesh2D::isCorner(int nodeID) const 
{
    return nodeID == 0 || nodeID == nx_ - 1 || nodeID == nx_ * (ny_ - 1) || nodeID == ny_ * nx_ - 1;
}

} // namespace