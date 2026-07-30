#ifndef STRUCTUREDMESH_HPP
#define STRUCTUREDMESH_HPP

#include <optional>
#include <unordered_map>
#include <string>

#include "Mesh2D.hpp"

namespace heat2d::mesh
{

struct Domain2D
{
    double left_;
    double right_;
    double bottom_;
    double top_;   
};

class StructuredMesh2D : public Mesh2D
{
    static const std::unordered_map<std::string, std::pair<int,int>> inward_directions_;

    private:
        Domain2D domain_;
        std::size_t nx_;
        std::size_t ny_;

    protected:
        void meshDomain() override;

    public:

        // Constructors
        StructuredMesh2D(double left, double right, double bottom, double top, std::size_t nx, std::size_t ny);
        StructuredMesh2D(const Domain2D& domain, std::size_t nx, std::size_t ny);
        
        // Getters
        inline std::size_t getNx() const {return nx_;};
        inline std::size_t getNy() const {return ny_;};
        inline double getDx() const {return (domain_.right_ - domain_.left_) / (nx_ - 1);};
        inline double getDy() const {return (domain_.top_ - domain_.bottom_) / (ny_ - 1);};
        inline double getMeshSize() const override {return std::min(getDx(), getDy());};
        inline const Domain2D& getDomain() const {return domain_;};
        std::optional<std::size_t> getNodeID(std::size_t i, std::size_t j) const; // from grid indices i and j
        std::optional<std::size_t> getNeighbor(std::size_t, const std::pair<int,int>&) const;
        inline const std::pair<int,int>& getBoundaryInwardDirection(const std::string& tag) const {return inward_directions_.at(tag);};

        // Specific helpers
        bool isCorner(std::size_t) const;
        double getElementArea(std::size_t elementID) const override;
        
};

};// namespace

#endif // ifndef STRUCTUREDMESH_HPP