#ifndef MESH_HPP
#define MESH_HPP

#include <unordered_map>
#include <stdexcept>
#include <vector>

namespace spatial
{

struct Node2D
{
    int nodeID_;
    double x_;
    double y_;
};

struct Domain2D
{
    double left_;
    double right_;
    double bottom_;
    double top_;   
};

enum class Side {Left, Right, Bottom, Top};

class Mesh2D
{
    protected:

        // nodes_ contains all nodes, inner_nodes_, boundary_nodes_ and boundaries_ contain IDs
        std::vector<Node2D> nodes_;
        std::vector<int> inner_nodes_;
        std::vector<int> boundary_nodes_;
        std::vector<bool> is_boundary_;
        std::unordered_map<Side, std::vector<int>> boundaries_;
        Domain2D domain_;

        // Populate the data members:
        virtual void meshDomain() = 0;

    public:

        // Constructors, assignments and destructor:
        Mesh2D(double left, double right, double bottom, double top);
        Mesh2D(const Domain2D& domain);

        Mesh2D(const Mesh2D&) = delete;
        Mesh2D& operator=(const Mesh2D&) = delete;
        Mesh2D(const Mesh2D&&) = delete;
        Mesh2D& operator=(const Mesh2D&&) = delete;

        virtual ~Mesh2D() = default;
        
        // Getters:
        inline const Domain2D& getDomain() const {return domain_;};
        inline const std::vector<Node2D>& getNodes() const {return nodes_;};
        inline const std::vector<int>& getInnerNodes() const {return inner_nodes_;};
        inline const std::vector<int>& getBoundaryNodes() const {return boundary_nodes_;};
        inline const std::unordered_map<Side, std::vector<int>>& getBoundaries() const {return boundaries_;};
        inline const Node2D& getNode(int nodeID) const {return nodes_[nodeID];}

        // Other helpers:
        inline bool isInner(int nodeID) const {return !is_boundary_[nodeID];};
        inline bool isBoundary(int nodeID) const {return is_boundary_[nodeID];};
};

}; // namespace

#endif // ifndef MESH_HPP