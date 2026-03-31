#ifndef SPATIALDISCRETIZATION2D_HPP
#define SPATIALDISCRETIZATION2D_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "BoundaryCondition.hpp"
#include "Mesh2D.hpp"

namespace spatial
{

// Unique pointer required for run-time polymorphism
using BoundaryConditions = std::unordered_map<DomainSide, std::unique_ptr<BoundaryCondition>>;

class SpatialDiscretization2D
{
    protected:
        const Mesh2D& mesh_;
        BoundaryConditions boundary_conditions_;
        std::function<double (double x, double y, double t)> source_;
        double alpha_;

        // Sparse matrix and tripletlist for assembly
        Eigen::SparseMatrix<double> matrix_;
        std::vector<Eigen::Triplet<double>> tripletList;
        Eigen::VectorXd b_;

        // Mappings for nodes in the local, reduced space (Dirichlet nodes are removed)
        std::vector<int> local_to_global_;
        std::vector<int> global_to_local_;

        // Check if node has prescribed Dirichlet BCs
        std::vector<bool> is_dirichlet_;

    public:
        SpatialDiscretization2D(double alpha, const Mesh2D& mesh, BoundaryConditions boundary_conditions, std::function<double (double, double, double)> source) 
        : alpha_(alpha),
        mesh_(mesh), 
        boundary_conditions_(boundary_conditions),
        source_(source),
        global_to_local_(mesh_.getNodes().size(), - 1),
        is_dirichlet_(mesh_.getNodes().size(), false) 
        {};

        virtual ~SpatialDiscretization2D() = default;

        // Discretize and build matrix A and vector b
        virtual void buildMappings() = 0;
        virtual void discretize() = 0;
        virtual void applyLaplacian() = 0;
        virtual void applyBoundaryConditions() = 0;
        virtual void updateRHS(double t) = 0;

        // Solves Au = b for steady-state problems. For time-dependent PDEs, this is unused.
        virtual Eigen::VectorXd solveSteadyState() = 0;
        virtual Eigen::VectorXd reduce(std::function<double (double, double)>) = 0;
        virtual Eigen::VectorXd fillDirichletNodes(const Eigen::Ref<const Eigen::VectorXd>&) = 0;

        // Getters
        inline const Eigen::SparseMatrix<double>& getMatrix() const {return matrix_;};
        inline const Eigen::VectorXd& getVector() const {return b_;};
};

}; // namespace
#endif // ifndef SPATIALDISCRETIZATION2D_HPP