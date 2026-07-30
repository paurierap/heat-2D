#include <Eigen/Sparse>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include "FiniteDifference2D.hpp"
#include "BoundaryConditions.hpp"
#include "StructuredMesh2D.hpp"

namespace heat2d::solver
{

FiniteDifference2D::FiniteDifference2D(std::function<double (double, double)> alpha, const mesh::StructuredMesh2D& mesh, BoundaryConditions boundary_conditions, std::function<double (double, double, double)> source)
: SpatialDiscretization2D(alpha, mesh, boundary_conditions, source), mesh_(mesh)
{
    // Precompute Dirichlet nodes and check if there are any Neumann BCs (helpful to determine if the Laplacian matrix is SPD)
    for (const auto& [tag, BC] : boundary_conditions_)
    {
        if (BC->getType() == bc::BoundaryConditionType::Dirichlet)
        {
            for (std::size_t nodeID : mesh_.getBoundary(tag)) is_dirichlet_[nodeID] = true;
        }
        else isMatrixSPD = false;
    }

    buildMappings();

    // Resize arrays for reduced space
    std::size_t local_space_size = local_to_global_.size();
    tripletList.reserve(5 * local_space_size);
    matrix_.resize(local_space_size, local_space_size);
    b_.resize(local_space_size);
}

// Create a mapping to reduce system size by omitting Dirichlet boundary conditions
void FiniteDifference2D::buildMappings()
{
   const std::vector<mesh::Node2D>& nodes = mesh_.getNodes();
    
    std::size_t free_index = 0;
    for (const auto& node : nodes)
    {
        std::size_t globalID = node.nodeID_;

        if (is_dirichlet_[globalID]) continue;

        global_to_local_[globalID] = free_index;
        local_to_global_.push_back(globalID);
        free_index++;
    } 
}

void FiniteDifference2D::discretize()
{
    std::cout << "\nDiscretizing the spatial domain using finite differences...\n";

    applyLaplacian();
    applyBoundaryConditions();
    matrix_.setFromTriplets(tripletList.begin(), tripletList.end());

    std::cout << "  -> Spatial discretization was successful.\n";
}

// Diagonal contribution to u_{i,j}
void FiniteDifference2D::addDiagonalTerm(std::size_t nodeID)
{
    std::size_t localID = global_to_local_[nodeID];
    double x = mesh_.getNode(nodeID).x_;
    double y = mesh_.getNode(nodeID).y_;
    double dx = mesh_.getDx();
    double dy = mesh_.getDy();

    tripletList.emplace_back(localID, localID, 
        -(alpha_(x + 0.5 * dx, y) + alpha_(x - 0.5 * dx, y)) / (dx*dx) 
        -(alpha_(x, y + 0.5 * dy) + alpha_(x, y - 0.5 * dy)) / (dy*dy));
}

// Off diagonal contributions (multiplier parameter, defaulted to 1.0, included in case there is a contribution from Neumann BCs)
void FiniteDifference2D::addOffDiagonalTerm(std::size_t nodeID, const std::pair<int, int>& direction, double multiplier)
{
    auto [dirx, diry] = direction;
    std::optional<std::size_t> neighbor = mesh_.getNeighbor(nodeID, direction);

    // Check if neighbor exists (in case of boundary nodes)
    if (!neighbor) return;

    // Check if neighbor has prescribed Dirichlet BCs
    if (is_dirichlet_[*neighbor]) return;

    std::size_t localID = global_to_local_[nodeID];
    std::size_t neighbor_local = global_to_local_[*neighbor];

    // Get coordinates of the node to evaluate alpha at the midpoint of the stencil
    double x = mesh_.getNode(nodeID).x_;
    double y = mesh_.getNode(nodeID).y_;

    // Horizontal nodes of the stencil
    if (dirx)
    {
        double dx = mesh_.getDx();
        tripletList.emplace_back(localID, neighbor_local, alpha_(x + 0.5 * dirx * dx, y) / (dx * dx) * multiplier);
        return;
    }
    
    // Vertical nodes of the stencil
    double dy = mesh_.getDy();
    tripletList.emplace_back(localID, neighbor_local, alpha_(x, y + 0.5 * diry * dy) / (dy * dy) * multiplier);
}

// Second order discretization approximation is applied to the inner nodes. If an inner node has a Dirichlet boundary node, this is later treated when applying boundary conditions.
void FiniteDifference2D::applyLaplacian()
{
    for (std::size_t globalID : mesh_.getInnerNodes())
    {
        // u_{i,j}
        addDiagonalTerm(globalID);
        
        // u_{i-1,j}, u_{i+1,j}, u_{i,j-1}, u_{i,j+1}
        for (const auto& direction : stencil) addOffDiagonalTerm(globalID, direction);
    }

    return;
}

// The contributions to the matrix A from the boundary conditions (mainly Neumann BC's) are here considered. Dirichlet BC's and the extra term in Neumann are treated separately in a vector b. This way, A is constant and computed only once at the beginning of execution.
void FiniteDifference2D::applyBoundaryConditions()
{
    // A boundary node can have 1 or 2 (corners) sides. If it belongs to a side with a Dirichlet BC, the node (and its row in A) is omitted. If it's a corner, a Dirichlet BC has preference over Neumann. If Neumann-Neumann, BCs are treated naturally.
    const std::vector<mesh::BoundaryNode2D>& boundary_nodes = mesh_.getBoundaryNodes();

    for (const auto& boundary_node : boundary_nodes)  
    {
        if (is_dirichlet_[boundary_node.nodeID_]) continue;
        applyFluxBoundaryCondition(boundary_node);            
    }

    return;
}

// Use ghost nodes, whereby the boundary node is treated almost like an inner node with a 4-point stencil (see https://www.12000.org/my_notes/neumman_BC/Neumman_BC.htm).
void FiniteDifference2D::applyFluxBoundaryCondition(const mesh::BoundaryNode2D& boundary_node)
{
    std::size_t globalID = boundary_node.nodeID_;
    std::size_t localID  = global_to_local_[globalID];
    double x = boundary_node.x_;
    double y = boundary_node.y_;

    addDiagonalTerm(globalID);

    for (auto [dirx, diry] : stencil)
    {
        bool isInward = false;
        const std::string* activeTag = nullptr;
        for (const auto& tag : boundary_node.tags_)
        {
            auto [inx, iny] = mesh_.getBoundaryInwardDirection(tag);
            if (inx == dirx && iny == diry) { isInward = true; activeTag = &tag; break; }
        }

        addOffDiagonalTerm(globalID, {dirx, diry}, isInward ? 2.0 : 1.0);

        if (isInward)
        {
            const bc::BoundaryCondition& BC = getBoundaryCondition(*activeTag);
            double h = dirx ? mesh_.getDx() : mesh_.getDy();
            double alpha_mid = dirx ? alpha_(x + 0.5*dirx*h, y) : alpha_(x, y + 0.5*diry*h);

            double correction = -2.0 * alpha_mid / h * BC.u_coeff(x,y) / BC.du_coeff(x,y);
            tripletList.emplace_back(localID, localID, correction);
        }
    }
}

void FiniteDifference2D::updateRHS(double t)
{
    b_.setZero();
    
    // A boundary node can have 1 or 2 (corners) sides. If it belongs to a side with a Dirichlet BC, the node (and its row in A) is omitted. If it's a corner, a Dirichlet BC has preference over Neumann. If Neumann-Neumann, BCs are treated naturally.
    const std::vector<mesh::BoundaryNode2D>& boundary_nodes = mesh_.getBoundaryNodes();
    for (const auto& boundary_node : mesh_.getBoundaryNodes()) 
    {
        if (is_dirichlet_[boundary_node.nodeID_]) updateDirichletBoundaryCondition(boundary_node, t);
        else updateFluxBoundaryCondition(boundary_node, t);
    }
    
    // Source term
    const auto& nodes = mesh_.getNodes();
    for (std::size_t globalID : local_to_global_)
    {
        std::size_t localID = global_to_local_[globalID];
        b_[localID] += source_(nodes[globalID].x_, nodes[globalID].y_, t);
    }

    return;
}

void FiniteDifference2D::updateDirichletBoundaryCondition(const mesh::BoundaryNode2D& boundary_node, double t)
{
    std::size_t globalID = boundary_node.nodeID_;
    double x = boundary_node.x_;
    double y = boundary_node.y_;

    for (const auto& tag : boundary_node.tags_)
    {
        // Get directions and values for the stencil
        auto [inx, iny] = mesh_.getBoundaryInwardDirection(tag);
        std::size_t neighbor_inward = *mesh_.getNeighbor(globalID, {inx, iny});

        // Check only for corner nodes with Dirichlet-Dirichlet BCs
        if (is_dirichlet_[neighbor_inward]) continue;

        std::size_t neighbor_local = global_to_local_[neighbor_inward];

        // Add contribution to the equation of the inward neighbor (corresponding to the row of that node in vector b)
        //double h = (inward_normal == DomainSide::Left || inward_normal == DomainSide::Right) ? mesh_.getDx() : mesh_.getDy();
        
        if (inx)
        {
            double h = mesh_.getDx();
            b_[neighbor_local] += alpha_(x + 0.5 * inx * h, y) / (h * h) * getBoundaryCondition(tag).f(x,y,t);
        }
        else
        {
            double h = mesh_.getDy();
            b_[neighbor_local] += alpha_(x, y + 0.5 * iny * h) / (h * h) * getBoundaryCondition(tag).f(x,y,t);
        }
    }

    return;
}

void FiniteDifference2D::updateFluxBoundaryCondition(const mesh::BoundaryNode2D& boundary_node, double t)
{
    std::size_t globalID = boundary_node.nodeID_;
    std::size_t localID  = global_to_local_[globalID];
    double x = boundary_node.x_;
    double y = boundary_node.y_;

    for (const auto& tag : boundary_node.tags_)
    {
        auto [inx, iny] = mesh_.getBoundaryInwardDirection(tag);
        double h = inx ? mesh_.getDx() : mesh_.getDy();
        double alpha_mid = inx ? alpha_(x + 0.5*inx*h, y) : alpha_(x, y + 0.5*iny*h);

        const bc::BoundaryCondition& BC = getBoundaryCondition(tag);
        b_[localID] += 2.0 * alpha_mid / h * BC.f(x,y,t) / BC.du_coeff(x,y,t);
    }
}

// Solve Poisson's equation, ie du/dt = 0.
Eigen::VectorXd FiniteDifference2D::solveSteadyState()
{
    std::cout << "\nSolving steady-state problem...\n";
    Eigen::VectorXd reduced_sol_ = solve_reduced();
    std::cout << "  -> Steady-state solution was successful!\n";
    
    return fillDirichletNodes(reduced_sol_, 0.0);
}

Eigen::VectorXd FiniteDifference2D::fillDirichletNodes(const Eigen::Ref<const Eigen::VectorXd>& reduced_solution, double t) const
{
    Eigen::VectorXd solution(mesh_.getNodes().size());

    // Fill solution with Dirichlet nodes
    const std::vector<mesh::Node2D>& nodes = mesh_.getNodes();
    for (const auto& node : nodes)
    {
        std::size_t globalID = node.nodeID_;

        if (!is_dirichlet_[globalID]) solution[globalID] = reduced_solution[global_to_local_[globalID]];
    }
    
    for (const auto& [tag, BC] : boundary_conditions_)
    {
        if (BC->getType() == bc::BoundaryConditionType::Dirichlet)
        {
            for (std::size_t globalID : mesh_.getBoundary(tag))
            {
                mesh::BoundaryNode2D boundary_node = mesh_.getBoundaryNode(globalID);
                double x = boundary_node.x_;
                double y = boundary_node.y_;

                solution[globalID] = BC->f(x,y,t);
            }
        }
    }

    return solution;
}

Eigen::VectorXd FiniteDifference2D::reduce(std::function<double (double, double)> u)
{
    std::size_t reduced_spacesize = local_to_global_.size();
    Eigen::VectorXd reduced_u(reduced_spacesize);

    for (std::size_t i = 0; i < reduced_spacesize; ++i)
    {
        std::size_t globalID = local_to_global_[i];
        const mesh::Node2D& node = mesh_.getNode(globalID);

        reduced_u[i] = u(node.x_, node.y_);
    }

    return reduced_u;
}

Eigen::VectorXd FiniteDifference2D::solve_reduced()
{
    Eigen::VectorXd reduced_sol_(local_to_global_.size());

    // Populate b_
    updateRHS();
    
    // Direct LDL^T factorization (only if A is SPD)
    if (isMatrixSPD)
    {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
        ldlt.compute(-matrix_);
        if (ldlt.info() != Eigen::Success) throw std::runtime_error("LDLT factorization failed\n");
        
        reduced_sol_ = ldlt.solve(b_);
        
        Eigen::VectorXd residual = (-matrix_) * reduced_sol_ - b_;
        if (residual.norm() / b_.norm() > 1e-10) throw std::runtime_error("LDLT solve residual too large");
    }
    else // Fall back to LU
    {
        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;

        lu.compute(-matrix_);
        if (lu.info() != Eigen::Success) throw std::runtime_error("LU factorization failed\n");
        
        reduced_sol_ = lu.solve(b_);
        
        Eigen::VectorXd residual = (-matrix_) * reduced_sol_ - b_;
        if (residual.norm() / b_.norm() > 1e-10) throw std::runtime_error("LU solve residual too large");
    }

    return reduced_sol_;
}
}; // namespace
