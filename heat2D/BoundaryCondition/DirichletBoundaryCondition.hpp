#ifndef DIRICHLETBOUNDARYCONDITION_HPP
#define DIRICHLETBOUNDARYCONDITION_HPP

#include "BoundaryCondition.hpp"

namespace spatial
{

class DirichletBoundaryCondition : public BoundaryCondition
{
    public: 
        DirichletBoundaryCondition(Side side, std::function<double (double, double, double)> f) 
        : BoundaryCondition(side, std::move(f)) {};

        void applyBoundaryCondition(const StructuredMesh2D&, std::vector<Eigen::Triplet<double>>&) const override {};    
};

} // namespace
#endif // ifndef DIRICHLETBOUNDARYCONDITION_HPP