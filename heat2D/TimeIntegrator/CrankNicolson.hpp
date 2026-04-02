#ifndef CRANKNICOLSON_HPP
#define CRANKNICOLSON_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <stdexcept>

#include "TimeIntegrator.hpp"

namespace temporal
{

class CrankNicolson : public TimeIntegrator
{
    private:
        Eigen::SparseMatrix<double> M_lhs_, M_rhs_;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> LUsolver_;
        bool isInitialized_ = false;
        
    public:
        CrankNicolson(double timestep) 
        : TimeIntegrator(timestep)
        {};

        void setUp(const spatial::SpatialDiscretization2D& sd) override
        {
            const Eigen::SparseMatrix<double>& A = sd.getMatrix();

            M_lhs_ = Eigen::SparseMatrix<double>(A.rows(), A.cols());
            M_lhs_.setIdentity();
            M_lhs_ -= 0.5 * timestep_ * A;

            M_rhs_ = Eigen::SparseMatrix<double>(A.rows(), A.cols());
            M_rhs_.setIdentity();
            M_rhs_ += 0.5 * timestep_ * A;

            LUsolver_.compute(M_lhs_);

            if (LUsolver_.info() != Eigen::Success) throw std::runtime_error("LU factorization for Crank Nicolson LHS failed\n");

            isInitialized_ = true;
        }

        void step(spatial::SpatialDiscretization2D& sd, double t, Eigen::VectorXd& u) const override
        {
            if (!isInitialized_) throw std::logic_error("\nStep function for Crank Nicolson time integration was used before SetUp.\n");

            sd.updateRHS(t);
            Eigen::VectorXd b = sd.getVector();

            sd.updateRHS(t + timestep_);
            b += sd.getVector();

            // No aliasing since M_rhs_ * u creates a new temporary vector
            u = LUsolver_.solve(M_rhs_ * u + 0.5 * timestep_ * b);

            if (LUsolver_.info() != Eigen::Success) throw std::runtime_error("CN solve failed");
        }

        // Virtual factory for timestep remainder operations. Note that the clone does not transfer precomputed matrices. The caller must invoke setUp() on the clone.
        std::unique_ptr<TimeIntegrator> cloneWithTimestep(double timestep) const override
        {
            return std::make_unique<CrankNicolson>(timestep);
        }

};

} // namespace
#endif // ifndef
