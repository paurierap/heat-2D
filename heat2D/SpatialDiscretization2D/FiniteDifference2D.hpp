#ifndef FINITEDIFFERENCE2D_HPP
#define FINITEDIFFERENCE2D_HPP

#include <Eigen/Dense>
#include <array>
#include <functional>
#include <string>

#include "BoundaryCondition.hpp"
#include "SpatialDiscretization2D.hpp"
#include "StructuredMesh2D.hpp"

namespace heat2d::solver
{

class FiniteDifference2D: public SpatialDiscretization2D
{
    private: 
        static constexpr std::array<std::pair<int, int>, 4> stencil{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

        // Structured mesh required for finite differences
        const mesh::StructuredMesh2D& mesh_;
        bool hasNeumann = false;

    public:
        FiniteDifference2D(std::function<double (double, double)>, const mesh::StructuredMesh2D&, BoundaryConditions, std::function<double (double, double, double)>);

        void buildMappings() override;

        void discretize() override;

        void addDiagonalTerm(int);
        void addOffDiagonalTerm(int, const std::pair<int, int>&, double = 1.0);
        void applyLaplacian() override;

        void applyBoundaryConditions() override;
        void applyNeumannBoundaryCondition(const mesh::BoundaryNode2D&);

        void updateRHS(double t=0.0) override;
        void updateDirichletBoundaryCondition(const mesh::BoundaryNode2D&, double t);
        void updateNeumannBoundaryCondition(const mesh::BoundaryNode2D&, double t);

        Eigen::VectorXd solveSteadyState() override;
        Eigen::VectorXd reduce(std::function<double (double, double)>) override;
        Eigen::VectorXd fillDirichletNodes(const Eigen::Ref<const Eigen::VectorXd>&, double) const override;
        Eigen::VectorXd solve_reduced();
        virtual bool isSPD() const override {return !hasNeumann;};
};

} // namespace
#endif // ifndef FINITEDIFFERENCE2D_HPP