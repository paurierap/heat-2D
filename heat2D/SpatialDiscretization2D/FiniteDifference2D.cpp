#include <Eigen/Sparse>
#include <unordered_map>
#include <vector>

#include "FiniteDifference2D.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "StructuredMesh2D.hpp"

namespace spatial
{

FiniteDifference2D::FiniteDifference2D(const StructuredMesh2D& mesh)
: SpatialDiscretization2D(mesh), mesh_(mesh) 
{
    tripletList.reserve(5 * mesh_.getNodes().size()); 
};

void FiniteDifference2D::discretize(const BCmap& BCs)
{
    applyLaplacian();
    applyBoundaryConditions(BCs);
    matrix_.setFromTriplets(tripletList.begin(), tripletList.end());
};

void FiniteDifference2D::applyLaplacian()
{
    double dx = mesh_.getDx(), dy = mesh_.getDy();
    int nx = mesh_.getNx(), ny = mesh_.getNy();
    std::vector<int> inner_nodes = mesh_.getInnerNodes();

    for (int nodeID : inner_nodes)
    {
        tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID - 1, 1./ (dx * dx))); // u_{i-1,j}
        tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID + 1, 1. / (dx * dx))); // u_{i+1,j}
        tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID - nx, 1. / (dy * dy))); // u_{i,j-1}
        tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID + nx, 1. / (dy * dy))); // u_{i,j+1}
        tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID, -2. / (dx * dx) -2. / (dy * dy))); // u_{i,j}
    }
};

void FiniteDifference2D::applyBoundaryConditions(const BCmap& BCs)
{
    
}

}; // namespace