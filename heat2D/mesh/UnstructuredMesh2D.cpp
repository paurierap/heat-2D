#include "UnstructuredMesh2D.hpp"

#include <gmsh.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace heat2d::mesh {

UnstructuredMesh2D::UnstructuredMesh2D(const std::string& gmshFile) : Mesh2D() {
  gmsh::initialize();
  try {
    gmsh::open(gmshFile);
    
    meshDomain();
    computeBoundaryEdges();
  } catch (...) {
    gmsh::finalize();
    throw std::invalid_argument("UnstructuredMesh2D: no valid file provided.");
  }
  gmsh::finalize();
}

void UnstructuredMesh2D::meshDomain() {

  // Get nodes and coordinates.
  std::vector<std::size_t> nodeTags;
  std::vector<double> coords, paramCoords;
  gmsh::model::mesh::getNodes(nodeTags, coords, paramCoords, -1, -1, false,
                              false);

  std::unordered_map<std::size_t, std::size_t> tagToLocal;
  tagToLocal.reserve(nodeTags.size());

  nodes_.resize(nodeTags.size());
  for (std::size_t i = 0; i < nodeTags.size(); ++i) {
    std::size_t localID = i;
    tagToLocal[nodeTags[i]] = localID;
    nodes_[localID] = Node2D{localID, coords[3 * i], coords[3 * i + 1]};
  }

  // Get boundary nodes and groupings.
  std::vector<std::pair<int, int>> physicalGroups;
  gmsh::model::getPhysicalGroups(physicalGroups, 1);

  for (const auto& [dim, tag] : physicalGroups) {
    std::string name;
    gmsh::model::getPhysicalName(dim, tag, name);

    std::vector<int> entityTags;
    gmsh::model::getEntitiesForPhysicalGroup(dim, tag, entityTags);

    std::vector<std::size_t>& groupNodes = boundary_groups_[name];

    for (std::size_t entityTag : entityTags) {
      std::vector<std::size_t> entityNodeTags;
      std::vector<double> entityCoords, entityParamCoords;

      // includeBoundary=true also grabs the curve's endpoint nodes,
      // so corner nodes shared between two tags get both tags.
      gmsh::model::mesh::getNodes(entityNodeTags, entityCoords,
                                  entityParamCoords, dim, entityTag, true,
                                  false);

      for (std::size_t nt : entityNodeTags) {
        std::size_t localID = tagToLocal.at(nt);

        if (node_to_boundary_node_.find(localID) ==
            node_to_boundary_node_.end()) {
          BoundaryNode2D bNode;
          bNode.nodeID_ = localID;
          bNode.x_ = nodes_[localID].x_;
          bNode.y_ = nodes_[localID].y_;
          bNode.tags_.push_back(name);

          node_to_boundary_node_[localID] = boundary_nodes_.size();
          boundary_nodes_.push_back(std::move(bNode));
        } else
          boundary_nodes_[node_to_boundary_node_.at(localID)].tags_.push_back(
              name);

        groupNodes.push_back(localID);
      }
    }
  }

  // Get inner nodes
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (isNodeInner(i)) inner_nodes_.push_back(i);
  }

  // Get elements
  std::vector<int> elementTypes;
  std::vector<std::vector<std::size_t>> elementTags, elementNodeTags;
  gmsh::model::mesh::getElements(elementTypes, elementTags, elementNodeTags, 2,
                                 -1);

  element_offsets_.push_back(0);
  for (std::size_t t = 0; t < elementTypes.size(); ++t) {
    std::string elementName;
    int elemDim, order, numNodesPerElement, numPrimNodes;
    std::vector<double> paramCoordTmp;
    gmsh::model::mesh::getElementProperties(elementTypes[t], elementName,
                                            elemDim, order, numNodesPerElement,
                                            paramCoordTmp, numPrimNodes);

    const auto& typeNodeTags = elementNodeTags[t];
    std::size_t numElems = typeNodeTags.size() / numNodesPerElement;

    for (std::size_t e = 0; e < numElems; ++e) {
      for (int n = 0; n < numNodesPerElement; ++n) {
        std::size_t gTag = typeNodeTags[e * numNodesPerElement + n];
        element_connectivity_.push_back(tagToLocal.at(gTag));
      }
      element_offsets_.push_back(element_connectivity_.size());
    }
  }
}

double UnstructuredMesh2D::getElementArea(std::size_t elementID) const {
  if (elementID >= element_offsets_.size() - 1)
    throw std::out_of_range("Invalid elementID.");

  double area = 0.0;
  Node2D p0 = nodes_[element_connectivity_[element_offsets_[elementID]]];
  Node2D p1 = nodes_[element_connectivity_[element_offsets_[elementID] + 1]];
  Node2D p2 = nodes_[element_connectivity_[element_offsets_[elementID] + 2]];

  return std::abs((p0.x_ * (p1.y_ - p2.y_) + p1.x_ * (p2.y_ - p0.y_) +
                   p2.x_ * (p0.y_ - p1.y_)) /
                  2.0);
}

}  // namespace heat2d::mesh