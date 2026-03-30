#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <stdexcept>

#include "HeatPDE2D.hpp"
#include "SpatialDiscretization2D.hpp"
#include "TimeIntegrator.hpp"

HeatPDE2D::HeatPDE2D(spatial::SpatialDiscretization2D& spatial_discretization, temporal::TimeIntegrator& time_integrator, double t_start, std::function<double (double, double)> u_start)
: spatial_discretization_(spatial_discretization),
time_integrator_(time_integrator),
t_start_(t_start),
u_start_(u_start)
{
    spatial_discretization_.discretize();

    u_end_ = spatial_discretization_.reduce(u_start);
    
    // Cache necessary matrices (depending on the time integration scheme)
    time_integrator_.setUp(spatial_discretization_);
};

void HeatPDE2D::integrate(double t_end)
{
    if (t_end <= t_start_) throw std::invalid_argument("T_end must be larger than t_start.");

    const double dt = time_integrator_.getTimestep();
    const int n_steps = static_cast<int>(std::floor((t_end - t_start_) / dt));
    double t = t_start_;

    while (t < t_end) 
    {
        time_integrator_.step(spatial_discretization_, t, u_end_);
        t += dt;
    }

    const double remainder = t_end - t;
    if (remainder > 0.0)
    {
        std::unique_ptr<temporal::TimeIntegrator> tail = time_integrator_.cloneWithTimestep(remainder);
        tail->setUp(spatial_discretization_);
        tail->step(spatial_discretization_, t, u_end_);
    }
}