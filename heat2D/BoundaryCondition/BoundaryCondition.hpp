#ifndef BOUNDARYCONDITION_HPP
#define BOUNDARYCONDITION_HPP

#include <functional>
#include <vector>

namespace spatial
{

enum class BoundaryConditionType {Dirichlet, Neumann};

class BoundaryCondition
{
    protected:
        std::function<double (double, double, double)> f_;

    public:
        BoundaryCondition(std::function<double (double, double, double)> f) 
        : f_(f) {}; 

        virtual ~BoundaryCondition() = default;
        
        double f(double x, double y, double t=0.0) const {return f_(x,y,t);};

        virtual BoundaryConditionType getType() const = 0;
};

} // namespace
#endif // ifndef BOUNDARYCONDITION_HPP