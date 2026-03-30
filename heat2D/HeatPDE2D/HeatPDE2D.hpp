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
        
        HeatPDE2D(spatial::SpatialDiscretization2D&, temporal::TimeIntegrator&, double, std::function<double (double, double)>);

        void integrate(double);

        // Getters
        Eigen::VectorXd getSolution() const {return spatial_discretization_.fillDirichletNodes(u_end_);};
};

#endif // HEATPDE_HPP