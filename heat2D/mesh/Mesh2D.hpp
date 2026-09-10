#ifndef MESH_HPP
#define MESH_HPP

#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

  // Boundary nodes are mapped to boundary_nodes_; inner nodes are absent.
  std::unordered_map<std::size_t, std::size_t> node_to_boundary_node_;

  // Contains boundary nodes for each boundary group (tag)
  std::unordered_map<std::string, std::vector<std::size_t>> boundary_groups_;

  // Boundary edges, keyed by boundary group tag.
  std::unordered_map<std::string, std::vector<std::pair<std::size_t, std::size_t>>>
      boundary_edges_;

  // Compressed Sparse Row (CSR) representation of elements in the mesh
  std::vector<std::size_t> element_connectivity_;
  std::vector<std::size_t> element_offsets_;

  virtual void meshDomain() = 0;

  // Derives, for each boundary group tag, the element edges whose two
  // endpoints both belong to the tag's boundary node group.
  inline void computeBoundaryEdges() {
    for (const auto& [tag, group] : boundary_groups_) {
      std::unordered_set<std::size_t> groupSet(group.begin(), group.end());
      std::set<std::pair<std::size_t, std::size_t>> edgeSet;

      for (std::size_t e = 0; e < getNumElements(); ++e) {
        const auto nodes = getElementNodes(e);
        std::size_t numNodes = nodes.size();
        for (std::size_t i = 0; i < numNodes; ++i) {
          std::size_t a = nodes[i];
          std::size_t b = nodes[(i + 1) % numNodes];
          if (groupSet.count(a) && groupSet.count(b)) {
            if (a > b) std::swap(a, b);
            edgeSet.emplace(a, b);
          }
        }
      }

      boundary_edges_[tag].assign(edgeSet.begin(), edgeSet.end());
    }
  };

 public:
  // Default constructor
  Mesh2D() = default;

  // Virtual destructor
  virtual ~Mesh2D() = default;

  // Copy and move constructors and assignment operators
  Mesh2D(const Mesh2D&) = default;
  Mesh2D& operator=(const Mesh2D&) = default;
  Mesh2D(Mesh2D&&) = default;
  Mesh2D& operator=(Mesh2D&&) = default;

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
    if (element_offsets_.empty() || elementID + 1 >= element_offsets_.size())
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

  // Compute pairs of nodes belonging to the same element lying on the given boundary tag.
  inline const std::vector<std::pair<std::size_t, std::size_t>>&
  getBoundaryEdgeNodes(const std::string& tag) const {
    return boundary_edges_.at(tag);
  };

  inline std::vector<Node2D> getNodesFromIDs(const std::vector<std::size_t>& nodeIDs) const {
    std::vector<Node2D> nodes;
    for (std::size_t id : nodeIDs) nodes.push_back(getNode(id));
    return nodes;
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
