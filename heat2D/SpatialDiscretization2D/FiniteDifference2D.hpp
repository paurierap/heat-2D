#ifndef FINITEDIFFERENCE2D_HPP
#define FINITEDIFFERENCE2D_HPP

#include <Eigen/Dense>
#include <functional>
#include <unordered_map>

#include "BoundaryCondition.hpp"
#include "SpatialDiscretization2D.hpp"
#include "StructuredMesh2D.hpp"

namespace spatial
{

class FiniteDifference2D: public SpatialDiscretization2D
{
    private: 

        // Structured mesh required for finite differences
        const StructuredMesh2D& mesh_;

    public:
        FiniteDifference2D(double, const StructuredMesh2D&, const BoundaryConditions&, std::function<double (double, double, double)>);

        void buildMappings() override;

        void discretize() override;

        void addDiagonalTerm(int);
        void addOffDiagonalTerm(int, DomainSide, double = 1.0);
        void applyLaplacian() override;

        void applyBoundaryConditions() override;
        void applyNeumannBoundaryCondition(const BoundaryNode2D&);

        void updateRHS(double t=0.0) override;
        void updateBoundaryConditions(double);
        void updateDirichletBoundaryCondition(const BoundaryNode2D&, double t);
        void updateNeumannBoundaryCondition(const BoundaryNode2D&, double t);

        void updateSource(double);

        Eigen::VectorXd solve() override;
        Eigen::VectorXd reduce(std::function<double (double, double)>) override;
        Eigen::VectorXd fillDirichletNodes(const Eigen::Ref<const Eigen::VectorXd>&) override;
        Eigen::VectorXd solve_reduced();
};

} // namespace
#endif // ifndef FINITEDIFFERENCE2D_HPP