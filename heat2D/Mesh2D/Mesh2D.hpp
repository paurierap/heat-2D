#ifndef MESH_HPP
#define MESH_HPP

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace heat2d::mesh
{

struct Node2D
{
    int nodeID_;
    double x_;
    double y_;
};

struct BoundaryNode2D : Node2D
{
    std::vector<std::string> tags_;
};

class Mesh2D
{
    protected:

        // Contains all nodes; nodeID_ must match the index in this vector.
        std::vector<Node2D> nodes_;

        // Contain IDs corresponding to inner nodes
        std::vector<int> inner_nodes_;

        // Contains all boundary nodes
        std::vector<BoundaryNode2D> boundary_nodes_;

        // Map ID from node to boundary node (-1 if inner node)
        std::vector<int> node_to_boundary_node_;

        // Contains boundary nodes for each boundary group (tag)
        std::unordered_map<std::string, std::vector<int>> boundary_groups_;

        // Compressed Sparse Row (CSR) representation of elements in the mesh
        std::vector<int> element_connectivity_; 
        std::vector<int> element_offsets_; 

        virtual void meshDomain() = 0;

    public:
        // Default constructor
        Mesh2D() = default;

        // Virtual destructor
        virtual ~Mesh2D() = default;
        
        // Getters
        inline const std::vector<Node2D>& getNodes() const {return nodes_;};
        inline const std::vector<BoundaryNode2D>& getBoundaryNodes() const {return boundary_nodes_;};
        inline const std::unordered_map<std::string, std::vector<int>>& getBoundaryGroups() const {return boundary_groups_;};
        inline std::vector<int> getElementNodes(int elementID) const 
        {
            std::vector<int> elementNodes;
            if (elementID < 0 || elementID >= static_cast<int>(element_offsets_.size()) - 1) throw std::out_of_range("Invalid elementID.");
            int start = element_offsets_[elementID];
            int end = element_offsets_[elementID + 1];
            for (int i = start; i < end; ++i) elementNodes.push_back(element_connectivity_[i]);
            return elementNodes;
        };
        inline const std::vector<int>& getElementConnectivity() const {return element_connectivity_;};
        inline const std::vector<int>& getElementOffsets() const {return element_offsets_;};
        inline int getNumElements() const {return static_cast<int>(element_offsets_.size()) - 1;};
        inline const std::vector<int>& getInnerNodes() const {return inner_nodes_;};
        inline const std::vector<int>& getBoundary(const std::string& tag) const {return boundary_groups_.at(tag);};
        inline const Node2D& getNode(int nodeID) const {return nodes_[nodeID];};
        inline const BoundaryNode2D& getBoundaryNode(int nodeID) const 
        {
            if (isNodeInner(nodeID)) throw std::invalid_argument("Selected node is not on the boundary.");
            return boundary_nodes_[node_to_boundary_node_[nodeID]];
        };
        virtual double getMeshSize() const = 0;
        virtual double getElementArea(int elementID) const = 0;

        // Other helpers
        inline bool isNodeInner(int nodeID) const {return node_to_boundary_node_[nodeID] == -1;};
        inline bool isNodeBoundary(int nodeID) const {return !isNodeInner(nodeID);};
};

}; // namespace

#endif // ifndef MESH_HPP
