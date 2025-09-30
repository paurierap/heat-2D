#ifndef FINITEDIFFERENCE2D_HPP
#define FINITEDIFFERENCE2D_HPP

#include "DirichletBoundaryCondition.hpp"
#include "SpatialDiscretization2D.hpp"
#include "StructuredMesh2D.hpp"

namespace spatial
{

class FiniteDifference2D: public SpatialDiscretization2D
{
    private: 
        const StructuredMesh2D& mesh_;

    public:
        FiniteDifference2D(const StructuredMesh2D& mesh);

        void discretize(const BCmap&) override;
        void applyLaplacian() override;
        void applyBoundaryConditions(const BCmap&) override;
};

} // namespace
#endif // ifndef FINITEDIFFERENCE2D_HPP