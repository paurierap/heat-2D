#ifndef TIMEINTEGRATOR_HPP
#define TIMEINTEGRATOR_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <stdexcept>

#include "SpatialDiscretization2D.hpp"

namespace temporal
{

class TimeIntegrator
{
    protected:
        double timestep_;

    public:
        TimeIntegrator(double timestep) 
        : timestep_(timestep)
        {
            if (timestep <= 0) throw std::invalid_argument("The timestep must be a positive number.");
        };
        
        virtual ~TimeIntegrator() = default;
        
        virtual void setUp(const spatial::SpatialDiscretization2D&) = 0;

        // Advances u by one timestep for du/dt = A*u + b
        virtual void step(spatial::SpatialDiscretization2D&, double, Eigen::VectorXd&) const = 0;
        virtual std::unique_ptr<TimeIntegrator> cloneWithTimestep(double) const = 0;

        // Getters
        inline double getTimestep() const {return timestep_;};
};

} // namespace
#endif // ifndef
