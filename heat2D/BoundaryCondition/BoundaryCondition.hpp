#ifndef BOUNDARYCONDITION_HPP
#define BOUNDARYCONDITION_HPP

#include <Eigen/Sparse>
#include <functional>
#include <vector>

#include "StructuredMesh2D.hpp"

namespace spatial
{

class BoundaryCondition
{
    protected:
        const Side side_;
        std::function<double (double, double, double)> f_;

    public:
        BoundaryCondition(Side side, std::function<double (double, double, double)> f) 
        : side_(side), f_(std::move(f)) {}; 

        virtual ~BoundaryCondition() = default;
        
        double f(double x, double y, double t) const {return f_(x,y,t);};

        inline const Side getSide() const {return side_;};

        virtual void applyBoundaryCondition(const StructuredMesh2D&, std::vector<Eigen::Triplet<double>>&) const = 0;
};

} // namespace
#endif // ifndef BOUNDARYCONDITION_HPP