#ifndef MESH_HPP
#define MESH_HPP

#include <map>
#include <stdexcept>
#include <vector>

namespace spatial
{

enum class DomainSide {Left, Right, Bottom, Top};

struct Domain2D
{
    double left_;
    double right_;
    double bottom_;
    double top_;   
};

struct Node2D
{
    int nodeID_;
    double x_;
    double y_;
};

struct BoundaryNode2D : Node2D
{
    std::vector<DomainSide> sides_;
};

class Mesh2D
{
    protected:

        Domain2D domain_;

        // Contains all nodes; nodeID_ must match the index in this vector.
        std::vector<Node2D> nodes_;

        // Contains all boundary nodes
        std::vector<BoundaryNode2D> boundary_nodes_;

        // Map ID from node to boundary node (-1 if inner node)
        std::vector<int> node_to_boundary_node;

        // Contain IDs corresponding to nodes
        std::vector<int> inner_nodes_;
        std::map<DomainSide, std::vector<int>> boundaries_;
        
        virtual void meshDomain() = 0;

    public:

        // Constructors
        Mesh2D(double left, double right, double bottom, double top);
        Mesh2D(const Domain2D& domain);

        virtual ~Mesh2D() = default;
        
        // Getters
        inline const Domain2D& getDomain() const {return domain_;};
        inline const std::vector<Node2D>& getNodes() const {return nodes_;};
        inline const std::vector<BoundaryNode2D>& getBoundaryNodes() const {return boundary_nodes_;};
        inline const std::vector<int>& getInnerNodes() const {return inner_nodes_;};
        inline const std::map<DomainSide, std::vector<int>>& getBoundaries() const {return boundaries_;};
        inline const Node2D& getNode(int nodeID) const {return nodes_[nodeID];};
        inline const BoundaryNode2D& getBoundaryNode(int nodeID) const 
        {
            if (isInner(nodeID)) throw std::invalid_argument("Selected node is not on the boundary.");
            return boundary_nodes_[node_to_boundary_node[nodeID]];
        };
        const std::pair<DomainSide, DomainSide> getBoundaryNormalDirections(DomainSide) const;
        const std::pair<DomainSide, DomainSide> getBoundaryTangentialDirections(DomainSide) const;
        virtual double getMeshSize() const = 0;

        // Other helpers
        inline bool isInner(int nodeID) const {return node_to_boundary_node[nodeID] == -1;};
        inline bool isBoundary(int nodeID) const {return !isInner(nodeID);};
};

}; // namespace

#endif // ifndef MESH_HPP
