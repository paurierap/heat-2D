#include <Eigen/Sparse>
#include <functional>
#include <iostream>
#include <optional>
#include <vector>

#include "FiniteDifference2D.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "StructuredMesh2D.hpp"

namespace spatial
{

FiniteDifference2D::FiniteDifference2D(std::function<double (double, double)> alpha, const StructuredMesh2D& mesh, BoundaryConditions boundary_conditions, std::function<double (double, double, double)> source)
: SpatialDiscretization2D(alpha, mesh, boundary_conditions, source), mesh_(mesh)
{

    // Precompute Dirichlet nodes structures
    for (const auto& [side, BC] : boundary_conditions_)
    {
        if (BC->getType() == BoundaryConditionType::Dirichlet)
        {
            for (int nodeID : mesh_.getBoundaries().at(side)) is_dirichlet_[nodeID] = true;
        }
    }

    buildMappings();

    // Resize arrays for reduced space
    int local_space_size = local_to_global_.size();
    tripletList.reserve(5 * local_space_size);
    matrix_.resize(local_space_size, local_space_size);
    b_.resize(local_space_size);
}

// Create a mapping to reduce system size by omitting Dirichlet boundary conditions
void FiniteDifference2D::buildMappings()
{
    const std::vector<Node2D>& nodes = mesh_.getNodes();
    
    int free_index = 0;
    for (int i = 0; i < nodes.size(); ++i)
    {
        int nodeID = nodes[i].nodeID_;

        if (is_dirichlet_[nodeID]) continue;

        global_to_local_[i] = free_index;
        local_to_global_.push_back(i);
        free_index++;
    }
}

void FiniteDifference2D::discretize()
{
    applyLaplacian();
    applyBoundaryConditions();
    matrix_.setFromTriplets(tripletList.begin(), tripletList.end());
}

// Diagonal contribution to u_{i,j}
void FiniteDifference2D::addDiagonalTerm(int nodeID)
{
    int localID = global_to_local_[nodeID];
    double x = mesh_.getNode(nodeID).x_;
    double y = mesh_.getNode(nodeID).y_;
    double dx = mesh_.getDx();
    double dy = mesh_.getDy();

    tripletList.emplace_back(localID, localID, -(alpha_(x + 0.5 * dx, y) + alpha_(x - 0.5 * dx, y)) / (dx*dx) -(alpha_(x, y + 0.5 * dy) + alpha_(x, y - 0.5 * dy)) / (dy*dy));
}

// Off diagonal contributions (multiplier parameter, defaulted to 1.0, included in case there is a contribution from Neumann BCs)
void FiniteDifference2D::addOffDiagonalTerm(int nodeID, DomainSide side, double multiplier)
{
    std::optional<int> neighbor = mesh_.getNeighbor(nodeID, side);

    // Check if neighbor exists (in case of boundary nodes)
    if (!neighbor) return;

    // Check if neighbor has prescribed Dirichlet BCs
    if (is_dirichlet_[*neighbor]) return;

    int localID = global_to_local_[nodeID];
    int neighbor_local = global_to_local_[*neighbor];

    double x = mesh_.getNode(nodeID).x_;
    double y = mesh_.getNode(nodeID).y_;

    // Horizontal nodes of the stencil
    if (side == DomainSide::Left || side == DomainSide::Right)
    {
        double dx = mesh_.getDx();
        double sign = (side == DomainSide::Left) ? -1 : 1;
        tripletList.emplace_back(localID, neighbor_local, alpha_(x + 0.5 * sign * dx, y) / (dx * dx) * multiplier);
        return;
    }
    
    // Vertical nodes of the stencil
    double dy = mesh_.getDy();
    double sign = (side == DomainSide::Bottom) ? -1 : 1;
    tripletList.emplace_back(localID, neighbor_local, alpha_(x, y + 0.5 * sign * dy) / (dy * dy) * multiplier);
}

// Second order discretization approximation is applied to the inner nodes. If an inner node has a Dirichlet boundary node, this is later treated when applying boundary conditions.
void FiniteDifference2D::applyLaplacian()
{
    const std::vector<int>& inner_node_IDs = mesh_.getInnerNodes();
    for (int globalID : inner_node_IDs)
    {
        // u_{i,j}
        addDiagonalTerm(globalID);

        // u_{i-1,j}
        addOffDiagonalTerm(globalID, DomainSide::Left);

        // u_{i+1,j}
        addOffDiagonalTerm(globalID, DomainSide::Right);

        // u_{i,j-1}
        addOffDiagonalTerm(globalID, DomainSide::Bottom);

        // u_{i,j+1}
        addOffDiagonalTerm(globalID, DomainSide::Top);
    }
}

// The contributions to the matrix A from the boundary conditions (mainly Neumann BC's) are here considered. Dirichlet BC's and the extra term in Neumann are treated separately in a vector b. This way, A is constant and computed only once at the beginning of execution.
void FiniteDifference2D::applyBoundaryConditions()
{
    // A boundary node can have 1 or 2 (corners) sides. If it belongs to a side with a Dirichlet BC, the node (and its row in A) is omitted. If it's a corner, a Dirichlet BC has preference over Neumann. If Neumann-Neumann, BCs are treated naturally.
    const std::vector<BoundaryNode2D>& boundary_nodes = mesh_.getBoundaryNodes();
    for (const auto& boundary_node : boundary_nodes)  
    {
        if (is_dirichlet_[boundary_node.nodeID_]) continue;
        applyNeumannBoundaryCondition(boundary_node);
    }
}

// Use ghost nodes, whereby the boundary node is treated almost like an inner node with a 4-point stencil (see https://www.12000.org/my_notes/neumman_BC/Neumman_BC.htm) with an extra contribution to the vector b. Ensure neighboring nodes are valid (for Neumann-Neumann BC corner treatment).
void FiniteDifference2D::applyNeumannBoundaryCondition(const BoundaryNode2D& boundary_node)
{
    int globalID = boundary_node.nodeID_;

    // u_{i,j}
    addDiagonalTerm(globalID);

    std::vector<DomainSide> sides = boundary_node.sides_;

    // Get directions for the stencil
    DomainSide inward_normal = mesh_.getBoundaryNormalDirections(sides[0]).second; // Only inward
    DomainSide tangent1 = mesh_.getBoundaryTangentialDirections(sides[0]).first;
    DomainSide tangent2 = mesh_.getBoundaryTangentialDirections(sides[0]).second;

    // Inward neighbors contributions
    addOffDiagonalTerm(globalID, inward_normal, 2.0);

    if (mesh_.isCorner(boundary_node.nodeID_))
    {
        // Handle Neumann-Neumman corner (one of the tangent directions will not find a node as it is a corner).
        addOffDiagonalTerm(globalID, tangent1, 2.0);
        addOffDiagonalTerm(globalID, tangent2, 2.0);
    }
    else
    {
        // Tangential neighbors contributions
        addOffDiagonalTerm(globalID, tangent1);
        addOffDiagonalTerm(globalID, tangent2);
    }
}

void FiniteDifference2D::updateRHS(double t)
{
    b_.setZero();
    updateBoundaryConditions(t);
    updateSource(t);
}

void FiniteDifference2D::updateBoundaryConditions(double t)
{
    // A boundary node can have 1 or 2 (corners) sides. If it belongs to a side with a Dirichlet BC, the node (and its row in A) is omitted. If it's a corner, a Dirichlet BC has preference over Neumann. If Neumann-Neumann, BCs are treated naturally.
    const std::vector<BoundaryNode2D>& boundary_nodes = mesh_.getBoundaryNodes();
    for (const auto& boundary_node : mesh_.getBoundaryNodes()) 
    {
        if (is_dirichlet_[boundary_node.nodeID_]) updateDirichletBoundaryCondition(boundary_node, t);  
        else updateNeumannBoundaryCondition(boundary_node, t);
    }
};

void FiniteDifference2D::updateDirichletBoundaryCondition(const BoundaryNode2D& boundary_node, double t)
{
    int globalID = boundary_node.nodeID_;
    double x = boundary_node.x_;
    double y = boundary_node.y_;

    for (auto side : boundary_node.sides_)
    {
        // Get directions and values for the stencil
        DomainSide inward_normal = mesh_.getBoundaryNormalDirections(side).second; // Only inward
        int neighbor_inward = *mesh_.getNeighbor(globalID, inward_normal);

        // Check only for corner nodes with Dirichlet-Dirichlet BCs
        if (is_dirichlet_[neighbor_inward]) continue;

        int neighbor_local = global_to_local_[neighbor_inward];

        // Add contribution to the equation of the inward neighbor (corresponding to the row of that node in vector b)
        //double h = (inward_normal == DomainSide::Left || inward_normal == DomainSide::Right) ? mesh_.getDx() : mesh_.getDy();
        
        double h;
        switch (inward_normal)
        {
            case DomainSide::Left:
                h = mesh_.getDx();
                b_[neighbor_local] += alpha_(x - 0.5 * h, y) / (h * h) * boundary_conditions_.at(side)->f(x,y,t);
                break;

            case DomainSide::Right:
                h = mesh_.getDx();
                b_[neighbor_local] += alpha_(x + 0.5 * h, y) / (h * h) * boundary_conditions_.at(side)->f(x,y,t);
                break;

            case DomainSide::Bottom:
                h = mesh_.getDy();
                b_[neighbor_local] += alpha_(x, y - 0.5 * h) / (h * h) * boundary_conditions_.at(side)->f(x,y,t);
                break;

            case DomainSide::Top:
                h = mesh_.getDy();
                b_[neighbor_local] += alpha_(x, y + 0.5 * h) / (h * h) * boundary_conditions_.at(side)->f(x,y,t);
                break;
        }
    }
}

void FiniteDifference2D::updateNeumannBoundaryCondition(const BoundaryNode2D& boundary_node, double t)
{
    int globalID = boundary_node.nodeID_;
    int localID = global_to_local_[globalID];
    double x = boundary_node.x_;
    double y = boundary_node.y_;
    double h;

    for (const auto& side : boundary_node.sides_)
    {
        if (side == DomainSide::Left || side == DomainSide::Right) h = mesh_.getDx();
        else h = mesh_.getDy();

        b_[localID] += 2. * alpha_(x,y) / h * boundary_conditions_.at(side)->f(x,y,t);
    }
}

void FiniteDifference2D::updateSource(double t)
{
    for (int globalID : local_to_global_)
    {
        int localID = global_to_local_[globalID];
        const Node2D& node = mesh_.getNode(globalID);

        b_[localID] += source_(node.x_, node.y_, t);
    }
}

// Solve Poisson's equation, ie du/dt = 0.
Eigen::VectorXd FiniteDifference2D::solveSteadyState()
{
    Eigen::VectorXd reduced_sol_ = solve_reduced();
    return fillDirichletNodes(reduced_sol_, 0.0);
}

Eigen::VectorXd FiniteDifference2D::fillDirichletNodes(const Eigen::Ref<const Eigen::VectorXd>& reduced_solution, double t) const
{
    Eigen::VectorXd solution(mesh_.getNodes().size());

    // Fill solution with Dirichlet nodes
    const std::vector<Node2D>& nodes = mesh_.getNodes();
    for (const auto& node : nodes)
    {
        int globalID = node.nodeID_;

        if (!is_dirichlet_[globalID]) solution[globalID] = reduced_solution[global_to_local_[globalID]];
    }
    
    for (const auto& [side, BC] : boundary_conditions_)
    {
        if (BC->getType() == BoundaryConditionType::Dirichlet)
        {
            for (int globalID : mesh_.getBoundaries().at(side))
            {
                BoundaryNode2D boundary_node = mesh_.getBoundaryNode(globalID);
                double x = boundary_node.x_;
                double y = boundary_node.y_;

                solution[globalID] = boundary_conditions_.at(side)->f(x,y,t); 
            }
        }
    }

    return solution;
}

Eigen::VectorXd FiniteDifference2D::reduce(std::function<double (double, double)> u)
{
    int reduced_spacesize = local_to_global_.size();
    Eigen::VectorXd reduced_u(reduced_spacesize);

    for (int i = 0; i < reduced_spacesize; ++i)
    {
        int globalID = local_to_global_[i];
        const Node2D& node = mesh_.getNode(globalID);

        reduced_u[i] = u(node.x_, node.y_);
    }
    
    return reduced_u;
}

Eigen::VectorXd FiniteDifference2D::solve_reduced()
{
    Eigen::VectorXd reduced_sol_(local_to_global_.size());

    // Populate b_
    updateRHS();

    // Check if there are any Neumann BCs
    bool hasNeumann = std::any_of(boundary_conditions_.begin(), boundary_conditions_.end(),
    [](const auto& pair){return pair.second->getType() == BoundaryConditionType::Neumann;});

    // Direct LDL^T factorization (only if A is SPD)
    if (!hasNeumann)
    {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
        ldlt.compute(-matrix_);
        if (ldlt.info() != Eigen::Success) throw std::runtime_error("LDLT factorization failed\n");
        reduced_sol_ = ldlt.solve(b_);
        
        if (ldlt.info() != Eigen::Success) throw std::runtime_error("LDLT solve failed\n");
        
        std::cout << "\nSolving with LDL^T factorization was successful!\n";
    }
    else // Fall back to LU
    {
        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;

        lu.compute(-matrix_);
        if (lu.info() != Eigen::Success) throw std::runtime_error("LU factorization failed\n");
        
        reduced_sol_ = lu.solve(b_);
        if (lu.info() != Eigen::Success) throw std::runtime_error("LU solve failed\n");

        std::cout << "\nSolving with LU factorization was successful!\n";
    }

    return reduced_sol_;
}
}; // namespace