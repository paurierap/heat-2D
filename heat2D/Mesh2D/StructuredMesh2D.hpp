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
        std::optional<int> getNeighbor(int, const std::pair<int,int>&) const;
        inline const std::pair<int,int>& getBoundaryInwardDirection(const std::string& tag) const {return inward_directions_.at(tag);};
        
        // Specific helpers
        bool isCorner(int) const;
        double getElementArea(int elementID) const override;
        
};

};// namespace

#endif // ifndef STRUCTUREDMESH_HPP