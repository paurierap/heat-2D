#ifndef SPATIALDISCRETIZATION2D_HPP
#define SPATIALDISCRETIZATION2D_HPP

#include <Eigen/Sparse>
#include <memory>
#include <unordered_map>
#include <vector>

#include "BoundaryCondition.hpp"
#include "Mesh2D.hpp"

namespace spatial
{

using BCmap = std::unordered_map<Side, std::unique_ptr<BoundaryCondition>>;

class SpatialDiscretization2D
{
    protected:
        const Mesh2D& mesh_;
        Eigen::SparseMatrix<double> matrix_;
        std::vector<Eigen::Triplet<double>> tripletList;

    public:
        SpatialDiscretization2D(const Mesh2D& mesh) 
        : mesh_(mesh), matrix_(mesh_.getNodes().size(), mesh_.getNodes().size()) {};

        virtual ~SpatialDiscretization2D() = default;

        virtual void discretize(const BCmap&) = 0;
        virtual void applyLaplacian() = 0;
        virtual void applyBoundaryConditions(const BCmap&) = 0;

        inline const Eigen::SparseMatrix<double>& getMatrix() const {return matrix_;};
};

}; // namespace
#endif // ifndef SPATIALDISCRETIZATION2D_HPP