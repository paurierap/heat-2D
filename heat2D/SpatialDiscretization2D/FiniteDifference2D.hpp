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
        Eigen::VectorXd b_;

    public:
        FiniteDifference2D(const StructuredMesh2D&, const BoundaryConditions&, std::function<double (double, double, double)>);

        void buildMappings() override;

        void discretize() override;

        void addDiagonalTerm(int);
        void addOffDiagonalTerm(int, DomainSide, double = 1.0);
        void applyLaplacian() override;

        void applyBoundaryConditions() override;
        void applyNeumannBoundaryCondition(const BoundaryNode2D&) override;
        void updateBoundaryConditions(double = 0.0) override;
        void updateDirichletBoundaryCondition(const BoundaryNode2D&, double) override;
        void updateNeumannBoundaryCondition(const BoundaryNode2D&, double) override;

        void updateSource(double = 0.0) override;

        Eigen::VectorXd solve() override;
        Eigen::VectorXd solve_reduced();

        // Getters
        inline const Eigen::VectorXd& getVector() const {return b_;};
};

} // namespace
#endif // ifndef FINITEDIFFERENCE2D_HPP