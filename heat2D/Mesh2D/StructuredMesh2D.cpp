#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "StructuredMesh2D.hpp"

namespace heat2d::mesh
{

const std::unordered_map<std::string, std::pair<int,int>> StructuredMesh2D::inward_directions_ = {
    {"Left",   { 1,  0}},
    {"Right",  {-1,  0}},
    {"Bottom", { 0,  1}},
    {"Top",    { 0, -1}}
};
    
StructuredMesh2D::StructuredMesh2D(double left, double right, double bottom, double top, std::size_t nx, std::size_t ny) 
: Mesh2D()
{
    if (left >= right || bottom >= top) throw std::invalid_argument("Inconsistent geometrical constraints. The left (or bottom) side cannot be equal or larger than the right (or top) side.");

    domain_ = {left, right, bottom, top};

    if (nx <= 1 || ny <= 1) throw std::invalid_argument("StructuredMesh2D: number of nodes must be positive and greater than 1.");
    
    nx_ = nx;
    ny_ = ny;

    std::cout << "\nCreating StructuredMesh2D with domain [" << left << ", " << right << "] x [" << bottom << ", " << top << "] and " << nx_ << " x " << ny_ << " nodes...\n";

    meshDomain();

    std::cout << "  -> StructuredMesh2D created with " << nodes_.size() << " nodes (" << inner_nodes_.size() << " inner, " << boundary_nodes_.size() << " boundary).\n";
};

StructuredMesh2D::StructuredMesh2D(const Domain2D& domain, std::size_t nx, std::size_t ny) 
: StructuredMesh2D(domain.left_, domain.right_, domain.bottom_, domain.top_, nx, ny) {};

void StructuredMesh2D::meshDomain()
{
    double dx = getDx(), dy = getDy();

    nodes_.reserve(nx_ * ny_);
    inner_nodes_.reserve((nx_ - 2) * (ny_ - 2));
    boundary_nodes_.reserve(2 * (nx_ + ny_) - 4);
    boundary_groups_["Left"].reserve(ny_);
    boundary_groups_["Right"].reserve(ny_);
    boundary_groups_["Bottom"].reserve(nx_);
    boundary_groups_["Top"].reserve(nx_);
    element_connectivity_.reserve((nx_ - 1) * (ny_ - 1) * 6);
    element_offsets_.reserve((nx_ - 1) * (ny_ - 1) * 2);

    std::size_t boundary_node = 0;
    element_offsets_.push_back(0);
    for (std::size_t row = 0; row < ny_; ++row)
    {
        for (std::size_t col = 0; col < nx_; ++col)
        {
            std::size_t nodeID = nx_ * row + col;
            double x = domain_.left_ + col * dx;
            double y = domain_.bottom_ + row * dy;
            nodes_.push_back(Node2D{nodeID, x, y});

            std::vector<std::string> boundary_tags;
            
            if (row < ny_ - 1 && col < nx_ - 1) 
            {
                element_connectivity_.push_back(nodeID);
                element_connectivity_.push_back(nodeID + 1);
                element_connectivity_.push_back(nodeID + nx_);
                element_offsets_.push_back(element_connectivity_.size());

                element_connectivity_.push_back(nodeID + 1);
                element_connectivity_.push_back(nodeID + nx_ + 1);
                element_connectivity_.push_back(nodeID + nx_);
                element_offsets_.push_back(element_connectivity_.size());
            }

            if (col == 0) 
            {
                boundary_groups_["Left"].push_back(nodeID);
                boundary_tags.push_back("Left");
            } 
            if (col == nx_ - 1) 
            {
                boundary_groups_["Right"].push_back(nodeID);
                boundary_tags.push_back("Right");
            } 
            if (row == 0) 
            {
                boundary_groups_["Bottom"].push_back(nodeID);
                boundary_tags.push_back("Bottom");
            } 
            if (row == ny_ - 1) 
            {
                boundary_groups_["Top"].push_back(nodeID);
                boundary_tags.push_back("Top");
            }

            if (boundary_tags.empty()) inner_nodes_.push_back(nodeID);
            else
            {
                node_to_boundary_node_[nodeID] = boundary_node;
                boundary_node++;
                boundary_nodes_.push_back(BoundaryNode2D{nodeID, x, y, boundary_tags});
            }
        }
    }

    for (std::size_t i = 0; i < nodes_.size(); ++i)
    {
        if (nodes_[i].nodeID_ != i)
        {
            throw std::logic_error("StructuredMesh2D: nodeID_ must match nodes_ index.");
        }
    }
};

std::optional<std::size_t> StructuredMesh2D::getNodeID(std::size_t i, std::size_t j) const
{
    if (i < 0 || i >= nx_ || j < 0 || j >= ny_) return std::nullopt;
    return j * nx_ + i;
};

std::optional<std::size_t> StructuredMesh2D::getNeighbor(std::size_t nodeID, const std::pair<int,int>& direction) const
{
    int di = direction.first;
    int dj = direction.second;
    int i = nodeID % nx_;
    int j = nodeID / nx_;

    if (i + di < 0 || i + di >= nx_ || j + dj < 0 || j + dj >= ny_) return std::nullopt;
    return static_cast<std::size_t>(j + dj) * nx_ + static_cast<std::size_t>(i + di);
};

bool StructuredMesh2D::isCorner(std::size_t nodeID) const 
{
    return nodeID == 0 || nodeID == nx_ - 1 || nodeID == nx_ * (ny_ - 1) || nodeID == ny_ * nx_ - 1;
}

double StructuredMesh2D::getElementArea(std::size_t elementID) const
{
    if (elementID >= element_offsets_.size() - 1) throw std::out_of_range("Invalid elementID.");

    return getDx() * getDy() * 0.5;
}

} // namespace