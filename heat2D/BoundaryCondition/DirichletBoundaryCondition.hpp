#ifndef DIRICHLETBOUNDARYCONDITION_HPP
#define DIRICHLETBOUNDARYCONDITION_HPP

#include "BoundaryCondition.hpp"

namespace spatial
{

class DirichletBoundaryCondition : public BoundaryCondition
{
    public: 
        DirichletBoundaryCondition(std::function<double (double, double, double)> f) 
        : BoundaryCondition(f) {};

        BoundaryConditionType getType() const override {return BoundaryConditionType::Dirichlet;};
};

} // namespace
#endif // ifndef DIRICHLETBOUNDARYCONDITION_HPP