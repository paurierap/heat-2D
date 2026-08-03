#ifndef MESH_HPP
#define MESH_HPP

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace heat2d::mesh {

struct Node2D {
  std::size_t nodeID_;
  double x_;
  double y_;
};

struct BoundaryNode2D : Node2D {
  std::vector<std::string> tags_;
};

class Mesh2D {
 protected:
  // Contains all nodes
  std::vector<Node2D> nodes_;

  // Contain IDs corresponding to inner nodes
  std::vector<std::size_t> inner_nodes_;

  // Contains all boundary nodes
  std::vector<BoundaryNode2D> boundary_nodes_;

  // Map ID from node to boundary node (-1 if inner node)
  std::unordered_map<std::size_t, std::size_t> node_to_boundary_node_;

  // Contains boundary nodes for each boundary group (tag)
  std::unordered_map<std::string, std::vector<std::size_t>> boundary_groups_;

  // Compressed Sparse Row (CSR) representation of elements in the mesh
  std::vector<std::size_t> element_connectivity_;
  std::vector<std::size_t> element_offsets_;

  virtual void meshDomain() = 0;

 public:
  // Default constructor
  Mesh2D() = default;

  // Virtual destructor
  virtual ~Mesh2D() = default;

  // Getters
  inline const std::vector<Node2D>& getNodes() const { return nodes_; };
  inline const std::vector<BoundaryNode2D>& getBoundaryNodes() const {
    return boundary_nodes_;
  };
  inline const std::unordered_map<std::string, std::vector<std::size_t>>&
  getBoundaryGroups() const {
    return boundary_groups_;
  };
  inline std::vector<std::size_t> getElementNodes(std::size_t elementID) const {
    std::vector<std::size_t> elementNodes;
    if (elementID >= element_offsets_.size() - 1)
      throw std::out_of_range("Invalid elementID.");
    std::size_t start = element_offsets_[elementID];
    std::size_t end = element_offsets_[elementID + 1];
    for (std::size_t i = start; i < end; ++i)
      elementNodes.push_back(element_connectivity_[i]);
    return elementNodes;
  };
  inline const std::vector<std::size_t>& getElementConnectivity() const {
    return element_connectivity_;
  };
  inline const std::vector<std::size_t>& getElementOffsets() const {
    return element_offsets_;
  };
  inline std::size_t getNumElements() const {
    return element_offsets_.size() - 1;
  };
  inline const std::vector<std::size_t>& getInnerNodes() const {
    return inner_nodes_;
  };
  inline const std::vector<std::size_t>& getBoundary(
      const std::string& tag) const {
    return boundary_groups_.at(tag);
  };
  inline const Node2D& getNode(std::size_t nodeID) const {
    return nodes_[nodeID];
  };
  inline const BoundaryNode2D& getBoundaryNode(std::size_t nodeID) const {
    if (isNodeInner(nodeID))
      throw std::invalid_argument("Selected node is not on the boundary.");
    return boundary_nodes_[node_to_boundary_node_.at(nodeID)];
  };
  virtual double getMeshSize() const = 0;
  virtual double getElementArea(std::size_t elementID) const = 0;

  // Other helpers
  inline bool isNodeInner(std::size_t nodeID) const {
    return node_to_boundary_node_.find(nodeID) == node_to_boundary_node_.end();
  };
  inline bool isNodeBoundary(std::size_t nodeID) const {
    return !isNodeInner(nodeID);
  };
};

};  // namespace heat2d::mesh

#endif  // ifndef MESH_HPP
