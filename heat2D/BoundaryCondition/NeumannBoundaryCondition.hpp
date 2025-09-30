#ifndef NEUMANNBOUNDARYCONDITION_HPP
#define NEUMANNBOUNDARYCONDITION_HPP

#include <Eigen/Sparse>
#include <functional>
#include <vector>

#include "BoundaryCondition.hpp"
#include "Mesh2D.hpp"

namespace spatial
{

class NeumannBoundaryCondition : public BoundaryCondition
{
    public: 
        NeumannBoundaryCondition(Side side, std::function<double (double, double, double)> f) 
        : BoundaryCondition(side, std::move(f)) {};

        void applyBoundaryCondition(const StructuredMesh2D&, std::vector<Eigen::Triplet<double>>&) const override;
};

} // namespace
#endif // ifndef NEUMANNBOUNDARYCONDITION_HPP