#ifndef NEUMANNBOUNDARYCONDITION_HPP
#define NEUMANNBOUNDARYCONDITION_HPP

#include "BoundaryCondition.hpp"

namespace spatial
{

class NeumannBoundaryCondition : public BoundaryCondition
{
    public: 
        NeumannBoundaryCondition(std::function<double (double, double, double)> f) 
        : BoundaryCondition(f) {};

        BoundaryConditionType getType() const override {return BoundaryConditionType::Neumann;};
};

} // namespace
#endif // ifndef NEUMANNBOUNDARYCONDITION_HPP