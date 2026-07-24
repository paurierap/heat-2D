#ifndef STRUCTUREDMESH_HPP
#define STRUCTUREDMESH_HPP

#include <optional>

#include "Mesh2D.hpp"

namespace mesh
{

enum class DomainSide {Left, Right, Bottom, Top};

static constexpr int sideToIndex(DomainSide side)
{
    switch (side)
    {
        case DomainSide::Left: return 0;
        case DomainSide::Right: return 1;
        case DomainSide::Bottom: return 2;
        case DomainSide::Top: return 3;
        default: throw std::invalid_argument("Invalid DomainSide value.");
    }
}

struct Domain2D
{
    double left_;
    double right_;
    double bottom_;
    double top_;   
};

class StructuredMesh2D : public Mesh2D
{
    private:
        Domain2D domain_;
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
        inline const Domain2D& getDomain() const {return domain_;};
        std::optional<int> getNodeID(int i, int j) const; // from grid indices i and j
        std::optional<int> getNeighbor(int, DomainSide) const;
        const std::pair<DomainSide, DomainSide> getBoundaryNormalDirections(DomainSide) const;
        const std::pair<DomainSide, DomainSide> getBoundaryTangentialDirections(DomainSide) const;

        // Specific helpers
        bool isCorner(int) const;
        double getElementArea(int elementID) const override;
};

};// namespace

#endif // ifndef STRUCTUREDMESH_HPP