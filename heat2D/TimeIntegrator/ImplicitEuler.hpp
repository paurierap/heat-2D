#ifndef IMPLICITEULER_HPP
#define IMPLICITEULER_HPP

#include <cassert>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>

#include "TimeIntegrator.hpp"

namespace temporal
{

class ImplicitEuler : public TimeIntegrator
{
    private:
        Eigen::SparseMatrix<double> M_lhs_;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> LUsolver_;
        bool isInitialized_ = false;

    public:
        ImplicitEuler(double timestep) 
        : TimeIntegrator(timestep)
        {};

        void setUp(const spatial::SpatialDiscretization2D& sd) override
        {
            const Eigen::SparseMatrix<double>& A = sd.getMatrix();

            M_lhs_ = Eigen::SparseMatrix<double>(A.rows(), A.cols());
            M_lhs_.setIdentity();
            M_lhs_ -= timestep_ * A;

            LUsolver_.compute(M_lhs_);

            if (LUsolver_.info() != Eigen::Success) throw std::runtime_error("LU factorization for Implicit Euler failed\n");

            isInitialized_ = true;
        }

        void step(spatial::SpatialDiscretization2D& sd, double t, Eigen::VectorXd& u) const override
        {
            if (!isInitialized_) throw std::logic_error("\nStep function for Implicit Euler time integration was used before SetUp.\n");

            sd.updateRHS(t + timestep_);
            const Eigen::VectorXd& b = sd.getVector();

            // Create temporary to avoid aliasing
            Eigen::VectorXd tmp = u + timestep_ * b;
            u = LUsolver_.solve(tmp);

            if (LUsolver_.info() != Eigen::Success) throw std::runtime_error("IE solve failed\n");
        };

        // Virtual factory for timestep remainder operations. Note that the clone does not transfer precomputed matrices. The caller must invoke setUp() on the clone.
        std::unique_ptr<TimeIntegrator> cloneWithTimestep(double timestep) const override
        {
            return std::make_unique<ImplicitEuler>(timestep);
        }
};

} // namespace
#endif // ifndef
