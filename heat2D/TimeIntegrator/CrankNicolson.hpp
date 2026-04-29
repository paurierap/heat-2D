#ifndef CRANKNICOLSON_HPP
#define CRANKNICOLSON_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "SpatialDiscretization2D.hpp"
#include "TimeIntegrator.hpp"

namespace temporal
{

class CrankNicolson : public TimeIntegrator
{
    private:
        SparseMatrixRM M_lhs_;
        SparseMatrixRM M_rhs_;

        // Direct solvers
        Eigen::SimplicialLDLT<SparseMatrixRM> SPDsolver_;
        Eigen::SparseLU<SparseMatrixRM> LUsolver_;

        // Iterative solvers
        Eigen::ConjugateGradient<SparseMatrixRM, Eigen::Lower|Eigen::Upper> CGsolver_;
        Eigen::BiCGSTAB<SparseMatrixRM> BiCGSTABsolver_;

        // Buffer vectors
        mutable Eigen::VectorXd b_;
        mutable Eigen::VectorXd rhs_;
        mutable Eigen::VectorXd residual_;

        bool useIterativeSolver_;
        bool isInitialized_ = false;
        bool isMatrixSPD_ = false;
        
    public:
        CrankNicolson(double timestep) 
        : TimeIntegrator(timestep)
        {};

        void setUp(const spatial::SpatialDiscretization2D& sd) override
        {
            const SparseMatrixRM& A = sd.getMatrix();
            isMatrixSPD_ = sd.isSPD();

            // Heuristic for iterative solver choice. This is a very rough estimate and should be tuned based on benchmarking results.
            useIterativeSolver_ = (timestep_ * A.rows() < 200. );

            b_.resize(A.rows());
            rhs_.resize(A.rows());
            
            M_lhs_ = SparseMatrixRM(A.rows(), A.cols());
            M_lhs_.setIdentity();
            M_lhs_ -= 0.5 * timestep_ * A;

            M_rhs_ = SparseMatrixRM(A.rows(), A.cols());
            M_rhs_.setIdentity();
            M_rhs_ += 0.5 * timestep_ * A;

            // Try an SPD factorization if the matrix is SPD
            if (isMatrixSPD_)
            {
                if (useIterativeSolver_)
                {
                    CGsolver_.setMaxIterations(1000);
                    CGsolver_.setTolerance(1e-10);

                    CGsolver_.compute(M_lhs_);
                    if (CGsolver_.info() != Eigen::Success) throw std::runtime_error("Conjugate Gradient factorization for Crank Nicolson LHS failed\n");
                }
                else
                {
                    SPDsolver_.compute(M_lhs_);
                    if (SPDsolver_.info() != Eigen::Success) throw std::runtime_error("LDLT factorization for Crank Nicolson LHS failed\n");
                }
            }
            else
            {
                if (useIterativeSolver_)
                {
                    BiCGSTABsolver_.setMaxIterations(1000);
                    BiCGSTABsolver_.setTolerance(1e-10);

                    BiCGSTABsolver_.compute(M_lhs_);
                    if (BiCGSTABsolver_.info() != Eigen::Success) throw std::runtime_error("BiCGSTAB factorization for Crank Nicolson LHS failed\n");
                }
                else
                {
                    LUsolver_.compute(M_lhs_);
                    if (LUsolver_.info() != Eigen::Success) throw std::runtime_error("LU factorization for Crank Nicolson LHS failed\n");
                }
            }

            isInitialized_ = true;
        }

        void step(spatial::SpatialDiscretization2D& sd, double t, Eigen::VectorXd& u) const override
        {
            if (u.size() != M_rhs_.cols()) throw std::invalid_argument("CrankNicolson::step: wrong u size");

            if (b_.size() != M_rhs_.rows() || rhs_.size() != M_rhs_.rows()) throw std::logic_error("CrankNicolson buffers not initialized");

            if (!isInitialized_) throw std::logic_error("\nStep function for Crank Nicolson time integration was used before SetUp.\n");

            sd.updateRHS(t);
            b_ = sd.getVector();

            sd.updateRHS(t + timestep_);
            b_ += sd.getVector();

            rhs_.noalias() = M_rhs_ * u;
            rhs_ += 0.5 * timestep_ * b_;
            
            if (isMatrixSPD_)
            {
                if (useIterativeSolver_)
                {
                    u = CGsolver_.solveWithGuess(rhs_, u);

                    residual_ = M_lhs_ * u;
                    residual_ -= rhs_;
                    const double rhs_norm = rhs_.norm();
                    if (rhs_norm > 0.0 && residual_.norm() / rhs_norm > 1e-10)  throw std::runtime_error("Conjugate Gradient solve residual too large");
                }
                else
                {
                    u = SPDsolver_.solve(rhs_);

                    residual_ = M_lhs_ * u;
                    residual_ -= rhs_;
                    const double rhs_norm = rhs_.norm();
                    if (rhs_norm > 0.0 && residual_.norm() / rhs_norm > 1e-10) throw std::runtime_error("LDLT solve residual too large");
                }
            }
            else
            {
                if (useIterativeSolver_)
                {
                    u = BiCGSTABsolver_.solveWithGuess(rhs_, u);

                    residual_ = M_lhs_ * u;
                    residual_ -= rhs_;
                    const double rhs_norm = rhs_.norm();
                    if (rhs_norm > 0.0 && residual_.norm() / rhs_norm > 1e-10) throw std::runtime_error("BiCGSTAB solve residual too large");
                }
                else
                {
                    u = LUsolver_.solve(rhs_);

                    residual_ = M_lhs_ * u;
                    residual_ -= rhs_;
                    const double rhs_norm = rhs_.norm();
                    if (rhs_norm > 0.0 && residual_.norm() / rhs_norm > 1e-10) throw std::runtime_error("LU solve residual too large");
                }
            }
        }

        // Virtual factory for timestep remainder operations. Note that the clone does not transfer precomputed matrices. The caller must invoke setUp() on the clone.
        std::unique_ptr<TimeIntegrator> cloneWithTimestep(double timestep) const override
        {
            return std::make_unique<CrankNicolson>(timestep);
        }

};

} // namespace
#endif // ifndef CRANKNICOLSON_HPP
