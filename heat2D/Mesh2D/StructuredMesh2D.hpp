#ifndef STRUCTUREDMESH_HPP
#define STRUCTUREDMESH_HPP

#include "Mesh2D.hpp"

#include <optional>

namespace spatial
{

class StructuredMesh2D : public Mesh2D
{
    private:
        int nx_;
        int ny_;

    protected:
        void meshDomain() override;

    public:

        // Constructors
        StructuredMesh2D(double left, double right, double bottom, double top, int nx, int ny);
        StructuredMesh2D(const Domain2D& domain, int nx, int ny);
        
        // Getters
        inline int getNx() const {return nx_;};
        inline int getNy() const {return ny_;};
        inline double getDx() const {return (domain_.right_ - domain_.left_) / (nx_ - 1);};
        inline double getDy() const {return (domain_.top_ - domain_.bottom_) / (ny_ - 1);};
        inline double getMeshSize() const override {return std::min(getDx(), getDy());};
        std::optional<int> getNodeID(int i, int j) const; // from grid indices i and j
        std::optional<int> getNeighbor(int, DomainSide) const;

        // Specific helpers
        bool isCorner(int) const;
};

};// namespace

#endif // ifndef STRUCTUREDMESH_HPP