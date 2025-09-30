#include "NeumannBoundaryCondition.hpp"

namespace spatial
{

void NeumannBoundaryCondition::applyBoundaryCondition(const StructuredMesh2D& mesh, std::vector<Eigen::Triplet<double>>& tripletList) const
{
    double dx = mesh.getDx(), dy = mesh.getDy();
    int nx = mesh.getNx(), ny = mesh.getNy();
            
    for (int nodeID : mesh.getBoundaries().at(side_)) 
    {
        tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID, -2. / (dx * dx) -2. / (dy * dy))); // u_{i,j}
        
        switch (side_)
        {
            case Side::Left:
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID + 1, 2. / (dx * dx))); // u_{i+1,j}
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID - nx, 1. / (dy * dy))); // u_{i,j-1}
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID + nx, 1. / (dy * dy))); // u_{i,j+1}
                break;

            case Side::Right:
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID - 1, 2. / (dx * dx))); // u_{i-1,j}
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID - nx, 1. / (dy * dy))); // u_{i,j-1}
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID + nx, 1. / (dy * dy))); // u_{i,j+1}
                break;
            
            case Side::Bottom:
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID + nx, 2. / (dy * dy))); // u_{i+1,j}
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID - 1, 1./ (dx * dx))); // u_{i-1,j}
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID + 1, 1. / (dx * dx))); // u_{i+1,j}
                break;
            
            case Side::Top:
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID - nx, 2. / (dy * dy))); // u_{i+1,j}
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID - 1, 1./ (dx * dx))); // u_{i-1,j}
                tripletList.push_back(Eigen::Triplet<double>(nodeID, nodeID + 1, 1. / (dx * dx))); // u_{i+1,j}
        }
    }
};

} // namespace