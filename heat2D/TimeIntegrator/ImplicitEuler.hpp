#ifndef IMPLICITEULER_HPP
#define IMPLICITEULER_HPP

#include <cassert>
#include <Eigen/Dense>
#include <Eigen/Sparse>
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

            if (LUsolver_.info() != Eigen::Success) throw std::runtime_error("LU factorization for Crank Nicolson LHS failed\n");

            isInitialized_ = true;
        }

        void step(spatial::SpatialDiscretization2D& sd, double t, Eigen::VectorXd& u) const override
        {
            assert(isInitialized_);

            sd.updateRHS(t + timestep_);
            const Eigen::VectorXd& b = sd.getVector();

            u = LUsolver_.solve(u + timestep_ * b);
        };

        // Virtual factory for timestep remainder operations.
        std::unique_ptr<TimeIntegrator> cloneWithTimestep(double timestep) const override
        {
            return std::make_unique<ImplicitEuler>(timestep);
        }
};

} // namespace
#endif // ifndef
