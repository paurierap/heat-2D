#ifndef HEATPDE2D_HPP
#define HEATPDE2D_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>

#include "SpatialDiscretization2D.hpp"
#include "TimeIntegrator.hpp"

class HeatPDE2D
{
    private:
        spatial::SpatialDiscretization2D& spatial_discretization_;
        temporal::TimeIntegrator& time_integrator_;
        
        // Initial conditions
        double t_start_;
        std::function<double (double, double)> u_start_;
        
        Eigen::VectorXd u_end_;

    public:
        HeatPDE2D(spatial::SpatialDiscretization2D& spatial_discretization, temporal::TimeIntegrator& time_integrator, double t_start, std::function<double (double, double)> u_start) 
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

        // Getter
        Eigen::VectorXd getSolution() const {return spatial_discretization_.fillDirichletNodes(u_end_);};

        // DefaultCallback generates a default dummy function that does nothing when no callback function is provided.
        struct DefaultCallback 
        {
            void operator()(double, const Eigen::VectorXd&) const noexcept {}
        };

        // The callback option is used to provide IO support to a user. This function is called every step using the current time t and the solution vector. This way, the user can choose if the solution is to be outputted and with which frequency by means of a lambda function.
        template <typename CallbackFunction = DefaultCallback>
        void integrate(double t_end, CallbackFunction&& callback = {})
        {
            if (t_end <= t_start_) throw std::invalid_argument("T_end must be larger than t_start.");

            const double dt = time_integrator_.getTimestep();
            const int n_steps = static_cast<int>(std::floor((t_end - t_start_) / dt));
            int step_count = 0;

            while (step_count < n_steps) 
            {
                time_integrator_.step(spatial_discretization_, t_start_ + step_count * dt, u_end_);
                step_count++;

                // For IO of the solution
                callback(t_start_ + step_count * dt, u_end_);
            }

            const double remainder = t_end - (t_start_ + dt * n_steps);
            if (remainder > 1e-10 * dt)
            {
                std::unique_ptr<temporal::TimeIntegrator> tail = time_integrator_.cloneWithTimestep(remainder);
                tail->setUp(spatial_discretization_);
                tail->step(spatial_discretization_, t_start_ + dt * n_steps, u_end_);

                // For IO of the solution
                callback(t_start_ + step_count * dt, u_end_);
            }

            t_start_ = t_end;
        }
};

#endif // HEATPDE_HPP