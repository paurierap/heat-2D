#ifndef EXPLICITEULER_HPP
#define EXPLICITEULER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#include "TimeIntegrator.hpp"

namespace temporal
{

class ExplicitEuler : public TimeIntegrator
{
    public:
        ExplicitEuler(double timestep) 
        : TimeIntegrator(timestep)
        {};
        
        void setUp(const spatial::SpatialDiscretization2D& sd) override {};

        void step(spatial::SpatialDiscretization2D& sd, double t, Eigen::VectorXd& u) const override
        {
            sd.updateRHS(t);

            const Eigen::SparseMatrix<double>& A = sd.getMatrix();
            const Eigen::VectorXd& b = sd.getVector();

            // Prevent aliasing from expression templating in Eigen using eval()
            u += (timestep_ * (A * u + b)).eval();
        };

        // Virtual factory for timestep remainder operations. Note that the clone does not transfer precomputed matrices. The caller must invoke setUp() on the clone.
        std::unique_ptr<TimeIntegrator> cloneWithTimestep(double timestep) const override
        {
            return std::make_unique<ExplicitEuler>(timestep);
        }
};

} // namespace
#endif // ifndef
