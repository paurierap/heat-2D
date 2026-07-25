#include "UnstructuredMesh2D.hpp"

#include <algorithm>
#include <cmath>
#include <gmsh.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <set>
#include <unordered_map>

namespace heat2d::mesh 
{

UnstructuredMesh2D::UnstructuredMesh2D(const std::string& gmshFile)
: Mesh2D()
{
    gmsh::initialize();
    try
    {
        gmsh::open(gmshFile);
        meshDomain();
    }
    catch (...)
    {
        gmsh::finalize(); 
        throw std::invalid_argument("UnstructuredMesh2D: no valid file provided.");
    }
    gmsh::finalize();
}

void UnstructuredMesh2D::meshDomain()
{
    constexpr double tol = 1e-10;
    constexpr int dim = 2;

    // Validate 2D triangle elements only
    std::vector<int> elemTypes;
    std::vector<std::vector<std::size_t>> elemTags, elemNodeTags;
    gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags, dim);

    if (elemTypes.empty()) throw std::invalid_argument("UnstructuredMesh2D: no 2D elements found in file.");

    // -------------------------------------------------------------------------
    // 3. Load all nodes, build tag -> 0-based index map
    // -------------------------------------------------------------------------
    std::vector<std::size_t> nodeTags;
    std::vector<double> coord, parametricCoord;
    gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord, -1, -1, false, false);

    const int numNodes = static_cast<int>(nodeTags.size());

    // Map tag -> position in coord array (Gmsh tag order)
    std::unordered_map<std::size_t, int> tagToCoordIdx;
    tagToCoordIdx.reserve(numNodes);
    for (int i = 0; i < numNodes; ++i)
        tagToCoordIdx[nodeTags[i]] = i;

    // Sort tags for deterministic, reproducible node ordering
    std::vector<std::size_t> sortedTags = nodeTags;
    std::sort(sortedTags.begin(), sortedTags.end());

    std::unordered_map<std::size_t, int> tagToIndex;
    tagToIndex.reserve(numNodes);

    nodes_.reserve(numNodes);
    node_to_boundary_node_.resize(numNodes, -1);

    for (int idx = 0; idx < numNodes; ++idx)
    {
        std::size_t tag = sortedTags[idx];
        int ci = tagToCoordIdx[tag];
        nodes_.push_back(Node2D{idx, coord[3*ci], coord[3*ci + 1]});
        tagToIndex[tag] = idx;
    }

    // -------------------------------------------------------------------------
    // 4. Populate triangle elements, fix winding to CCW
    // -------------------------------------------------------------------------
    const auto& triNodeTags = elemNodeTags[0];
    const int numElems = static_cast<int>(triNodeTags.size()) / 3;
    elements_.reserve(numElems);

    for (int e = 0; e < numElems; ++e)
    {
        int n0 = tagToIndex[triNodeTags[3*e    ]];
        int n1 = tagToIndex[triNodeTags[3*e + 1]];
        int n2 = tagToIndex[triNodeTags[3*e + 2]];
        elements_.push_back({n0, n1, n2});
    }

    for (auto& el : elements_)
    {
        const auto& p0 = nodes_[el[0]];
        const auto& p1 = nodes_[el[1]];
        const auto& p2 = nodes_[el[2]];
        double cross = (p1.x_ - p0.x_) * (p2.y_ - p0.y_)
                     - (p1.y_ - p0.y_) * (p2.x_ - p0.x_);
        if (cross < 0.0) std::swap(el[1], el[2]); // flip to CCW
    }

    // -------------------------------------------------------------------------
    // 5. Identify boundary nodes by coordinate proximity to bounding box edges
    //    This also validates axis-alignment: any boundary node not on one of 
    //    the four edges implies a non-rectangular domain
    // -------------------------------------------------------------------------

    // Collect boundary node tags from 1D (edge) elements so we only check
    // nodes that Gmsh itself considers to be on the boundary
    std::vector<int> lineTypes;
    std::vector<std::vector<std::size_t>> lineTags, lineNodeTagsVec;
    gmsh::model::mesh::getElements(lineTypes, lineTags, lineNodeTagsVec, 1);

    std::set<std::size_t> boundaryTagSet;
    if (!lineTypes.empty())
        for (auto tag : lineNodeTagsVec[0])
            boundaryTagSet.insert(tag);

    boundaries_[sideToIndex(DomainSide::Left  )].reserve(32);
    boundaries_[sideToIndex(DomainSide::Right )].reserve(32);
    boundaries_[sideToIndex(DomainSide::Bottom)].reserve(32);
    boundaries_[sideToIndex(DomainSide::Top   )].reserve(32);

    int boundaryCounter = 0;
    for (int idx = 0; idx < numNodes; ++idx)
    {
        std::size_t tag = sortedTags[idx];
        if (!boundaryTagSet.count(tag))
        {
            inner_nodes_.push_back(idx);
            continue;
        }

        double x = nodes_[idx].x_, y = nodes_[idx].y_;
        std::vector<DomainSide> sides;

        bool onLeft   = std::abs(x - xmin) < tol;
        bool onRight  = std::abs(x - xmax) < tol;
        bool onBottom = std::abs(y - ymin) < tol;
        bool onTop    = std::abs(y - ymax) < tol;

        if (!onLeft && !onRight && !onBottom && !onTop)
            throw std::invalid_argument(
                "UnstructuredMesh2D: boundary node at (" + std::to_string(x) + ", " +
                std::to_string(y) + ") does not lie on an axis-aligned edge — "
                "domain is not a rectangle.");

        if (onLeft  ) sides.push_back(DomainSide::Left  );
        if (onRight ) sides.push_back(DomainSide::Right );
        if (onBottom) sides.push_back(DomainSide::Bottom);
        if (onTop   ) sides.push_back(DomainSide::Top   );

        for (DomainSide side : sides)
            boundaries_[sideToIndex(side)].push_back(idx);

        boundary_nodes_.push_back(BoundaryNode2D{idx, x, y, sides});
        node_to_boundary_node_[idx] = boundaryCounter++;
    }

    // -------------------------------------------------------------------------
    // 6. Compute mesh size as minimum edge length across all triangles
    // -------------------------------------------------------------------------
    meshSize_ = std::numeric_limits<double>::max();

    auto edgeLen = [&](int a, int b)
    {
        double dx = nodes_[a].x_ - nodes_[b].x_;
        double dy = nodes_[a].y_ - nodes_[b].y_;
        return std::sqrt(dx*dx + dy*dy);
    };

    for (const auto& el : elements_)
        meshSize_ = std::min({meshSize_,
                              edgeLen(el[0], el[1]),
                              edgeLen(el[1], el[2]),
                              edgeLen(el[0], el[2])});
}

bool UnstructuredMesh2D::isCorner(int nodeID) const
{
    if (node_to_boundary_node_[nodeID] == -1) return false;
    return boundary_nodes_[node_to_boundary_node_[nodeID]].sides_.size() == 2;
}

double UnstructuredMesh2D::getElementArea(int elementID) const
{
    const std::array<int, 3>& el = elements_[elementID];
    Node2D p0 = nodes_[el[0]];
    Node2D p1 = nodes_[el[1]];
    Node2D p2 = nodes_[el[2]];

    return 0.5 * std::abs((p1.x_ - p0.x_) * (p2.y_ - p0.y_) - (p1.y_ - p0.y_) * (p2.x_ - p0.x_));
}

} // namespace heat2d::mesh