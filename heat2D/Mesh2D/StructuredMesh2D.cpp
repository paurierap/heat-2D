#include "StructuredMesh2D.hpp"

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
    is_boundary_.reserve(nx_ * ny_);
    boundaries_[Side::Left].reserve(ny_);
    boundaries_[Side::Right].reserve(ny_);
    boundaries_[Side::Bottom].reserve(nx_);
    boundaries_[Side::Top].reserve(nx_);

    for (int row = 0; row < ny_; ++row) 
    {
        for (int col = 0; col < nx_; ++col) 
        {
            int nodeID = nx_ * row + col;
            nodes_.push_back(Node2D{nodeID, domain_.left_ + col * dx, domain_.bottom_ + row * dy});

            bool isBoundary = false;

            if (col == 0) 
            {
                boundaries_[Side::Left].push_back(nodeID);
                isBoundary = true;
            }
            if (col == nx_ - 1) 
            {
                boundaries_[Side::Right].push_back(nodeID);
                isBoundary = true;
            }
            if (row == 0) 
            {
                boundaries_[Side::Bottom].push_back(nodeID);
                isBoundary = true;
            }
            if (row == ny_ - 1) 
            {
                boundaries_[Side::Top].push_back(nodeID);
                isBoundary = true;
            }

            is_boundary_[nodeID] = isBoundary;
            if (isBoundary) boundary_nodes_.push_back(nodeID);
            else inner_nodes_.push_back(nodeID);
        }
    }

};

} // namespace