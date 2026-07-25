#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Mesh2D.hpp"
#include "StructuredMesh2D.hpp"

#ifdef HEAT2D_HAS_GMSH
#include "UnstructuredMesh2D.hpp"
#endif

using namespace heat2d::mesh;

// =============================================================================
// Fixture - Avoids repeated allocation
// =============================================================================
class StructuredMesh2DTest : public testing::Test
{
    protected: 
        static constexpr int nx = 11, ny = 21;
        static constexpr double left = 0, right = 1, bottom = 0, top = 1;
        std::vector<Node2D> nodes;
        const StructuredMesh2D mesh{left,right,bottom,top,nx,ny};
};

// =============================================================================
// Test 1 - Check constructor from invalid dimensions
// =============================================================================
TEST(StructuredMesh2D, InvalidDimensionsThrow) 
{
    EXPECT_THROW(StructuredMesh2D(0,1,0,-1,10,10), std::invalid_argument);
    EXPECT_THROW(StructuredMesh2D(0,-1,0,1,10,10), std::invalid_argument);
}

// =============================================================================
// Test 2 - Check constructor from invalid (negative) number of nodes
// =============================================================================
TEST(StructuredMesh2D, InvalidNumberOfNodesThrow) 
{
    EXPECT_THROW(StructuredMesh2D(0,1,0,1,-10,10), std::invalid_argument);
    EXPECT_THROW(StructuredMesh2D(0,1,0,1,10,-10), std::invalid_argument);
}

// =============================================================================
// Test 3 - Check constructor
// =============================================================================
TEST_F(StructuredMesh2DTest, ConstructsMeshFromDomainSides) 
{
    const StructuredMesh2D mesh(0,1,0,1,nx,ny);

    EXPECT_DOUBLE_EQ(mesh.getDx(), 1. / (nx - 1));
    EXPECT_DOUBLE_EQ(mesh.getDy(), 1. / (ny - 1));
}

// =============================================================================
// Test 4 - Check constructor using Domain struct
// =============================================================================
TEST_F(StructuredMesh2DTest, ConstructsMeshFromDomain) 
{
    const Domain2D domain{0, 1, 0, 1};
    const StructuredMesh2D mesh(domain,nx,ny);

    EXPECT_DOUBLE_EQ(mesh.getDx(), (domain.right_ - domain.left_) / (nx - 1));
    EXPECT_DOUBLE_EQ(mesh.getDy(), (domain.top_ - domain.bottom_) / (ny - 1));
}

// =============================================================================
// Test 5 - Check dx & dy calculations
// =============================================================================
TEST_F(StructuredMesh2DTest, DxDyCalculation) 
{
    EXPECT_DOUBLE_EQ(mesh.getDx(), (mesh.getDomain().right_ - mesh.getDomain().left_) / (nx - 1));
    EXPECT_DOUBLE_EQ(mesh.getDy(), (mesh.getDomain().top_ - mesh.getDomain().bottom_) / (ny - 1));
}

// =============================================================================
// Test 6 - Check appropriate counting of nodes
// =============================================================================
TEST_F(StructuredMesh2DTest, NodeCounting)
{
    EXPECT_EQ(mesh.getNx(), nx);
    EXPECT_EQ(mesh.getNy(), ny);
    EXPECT_EQ(mesh.getNodes().size(), mesh.getNx() * mesh.getNy());
    EXPECT_EQ(mesh.getBoundaryNodes().size(), 2 * (nx + ny) - 4);
    EXPECT_EQ(mesh.getBoundaryNodes().size() + mesh.getInnerNodes().size(), mesh.getNodes().size());
    EXPECT_EQ(mesh.getNodeID(6, 12), 12 * nx + 6);
}

// =============================================================================
// Test 7 - Check out of bounds nodes
// =============================================================================
TEST_F(StructuredMesh2DTest, NodeOutOfBounds)
{
    EXPECT_FALSE(mesh.getNodeID(-1, 1));
    EXPECT_FALSE(mesh.getNodeID(1, -1));
    EXPECT_FALSE(mesh.getNodeID(nx, 1));
    EXPECT_FALSE(mesh.getNodeID(1, ny));
}

// =============================================================================
// Test 8 - Check isCorner() function
// =============================================================================
TEST_F(StructuredMesh2DTest, CheckCorners)
{
    const std::vector<BoundaryNode2D>& t_corners = mesh.getBoundaryNodes();
    std::unordered_set<int> corners{0, 10, 220, 230};
    
    for (auto node : t_corners)
    {
        if (node.tags_.size() == 2)
        {
            EXPECT_TRUE(corners.count(node.nodeID_)) << "Node: " << node.nodeID_ << " is incorrectly classified as a corner node.";
            corners.erase(node.nodeID_);
        }
    }
    EXPECT_TRUE(corners.empty());

    EXPECT_TRUE(mesh.isCorner(0));
    EXPECT_TRUE(mesh.isCorner(10));
    EXPECT_TRUE(mesh.isCorner(220));
    EXPECT_TRUE(mesh.isCorner(230));
    EXPECT_FALSE(mesh.isCorner(231));
}

// =============================================================================
// Test 9 - Check correct grid generation
// =============================================================================
TEST_F(StructuredMesh2DTest, MeshGeneration)
{
    double dx = mesh.getDx(), dy = mesh.getDy();

    for (int row = 0; row < mesh.getNy(); ++row) 
    {
        for (int col = 0; col < mesh.getNx(); ++col) 
        {
            int nodeID = row * mesh.getNx() + col;
            EXPECT_NEAR(mesh.getNode(nodeID).x_, mesh.getDomain().left_ + col*dx, 1e-12);
            EXPECT_NEAR(mesh.getNode(nodeID).y_, mesh.getDomain().bottom_ + row*dy, 1e-12);
        }
    }
}

// =============================================================================
// Test 10 - Check correct size of sides in getBoundary()
// =============================================================================
TEST_F(StructuredMesh2DTest, BoundaryNodesAssignation)
{
    const std::vector<int>& left_boundary = mesh.getBoundary("Left");
    const std::vector<int>& right_boundary = mesh.getBoundary("Right");
    const std::vector<int>& bottom_boundary = mesh.getBoundary("Bottom");
    const std::vector<int>& top_boundary = mesh.getBoundary("Top");

    EXPECT_EQ(left_boundary.size(), ny);
    EXPECT_EQ(right_boundary.size(), ny);
    EXPECT_EQ(bottom_boundary.size(), nx);
    EXPECT_EQ(top_boundary.size(), nx);
}

// =============================================================================
// Test 11 - Check that getInnerNodes() and getBoundaryNodes() return, 
//           respectively, only inner and boundary nodes 
// =============================================================================
TEST_F(StructuredMesh2DTest, InnerBoundaryNodesSeparation)
{
    for (int nodeID : mesh.getInnerNodes()) 
    {
        EXPECT_FALSE(mesh.isNodeBoundary(nodeID)) << "Node: " << nodeID << " is incorrectly classified as a boundary node.";
    }

    for (auto node : mesh.getBoundaryNodes()) 
    {
        EXPECT_TRUE(mesh.isNodeBoundary(node.nodeID_)) << "Node: " << node.nodeID_ << " is incorrectly classified as an inner node.";
    }
}

// =============================================================================
// Test 12 - Check element count and node indices are valid 
// =============================================================================
TEST_F(StructuredMesh2DTest, ElementCountingAndNodeIndices)
{
    const std::vector<int>& element_connectivity = mesh.getElementConnectivity();
    const std::vector<int>& element_offsets = mesh.getElementOffsets();
    int numNodes = mesh.getNodes().size();
    int numElements = mesh.getNumElements();

    EXPECT_EQ(static_cast<int>(element_offsets.size()) - 1, (nx - 1) * (ny - 1) * 2); 

    for (int i = 0; i < numElements; ++i) 
    {
        int start = element_offsets[i];
        int end = element_offsets[i + 1];
        EXPECT_EQ(end - start, 3) << "Element " << i << " does not have 3 nodes.";
        
        for (int j = start; j < end; ++j) 
        {
            int nodeID = element_connectivity[j];
            EXPECT_GE(nodeID, 0) << "Element " << i << " has invalid node index: " << nodeID;
            EXPECT_LT(nodeID, numNodes) << "Element " << i << " has invalid node index: " << nodeID;
        }
    }
}

// =============================================================================
// Test 13 - Check that all element node indices are distinct
// =============================================================================
TEST_F(StructuredMesh2DTest, ElementNodesAreDistinct)
{
    const std::vector<int>& element_connectivity = mesh.getElementConnectivity();
    const std::vector<int>& element_offsets = mesh.getElementOffsets();
    int numElements = mesh.getNumElements();

    for (int i = 0; i < numElements; ++i) 
    {
        int start = element_offsets[i];
        int end = element_offsets[i + 1];

        std::unordered_set<int> seen;
        for (int j = 0; j < end - start; ++j)
        {
            int nodeID = element_connectivity[start + j];
            EXPECT_FALSE(seen.count(nodeID)) << "Element " << i << " has duplicate node: " << nodeID;
            seen.insert(nodeID);
        }
    }
}

// =============================================================================
// Test 14 - Check element areas are positive and sum to domain area
// =============================================================================
TEST_F(StructuredMesh2DTest, ElementAreasPositiveAndSumToDomainArea)
{
    double totalArea = 0.0;
    int numElements = mesh.getNumElements();

    for (int i = 0; i < numElements; ++i)
    {
        double area = mesh.getElementArea(i);

        EXPECT_GT(area, 0.0) << "Element " << i << " has non-positive area";
        totalArea += area;
    }

    double domainArea = (right - left) * (top - bottom);
    EXPECT_NEAR(totalArea, domainArea, 1e-10);
}

// =============================================================================
// Test 15 - Check that every node belongs to at least one element
// =============================================================================
TEST_F(StructuredMesh2DTest, AllNodesCoveredByElements)
{
    const std::vector<int>& element_connectivity = mesh.getElementConnectivity();
    const std::vector<int>& element_offsets = mesh.getElementOffsets();
    int numNodes = mesh.getNodes().size();
    int numElements = mesh.getNumElements();
    std::vector<bool> seen(numNodes, false);

    for (int i = 0; i < numElements; ++i)
    {
        int start = element_offsets[i];
        int end = element_offsets[i + 1];

        for (int j = start; j < end; ++j)
        {
            int nodeID = element_connectivity[j];
            seen[nodeID] = true;
        }
    }
        
    for (int i = 0; i < numNodes; ++i) EXPECT_TRUE(seen[i]) << "Node " << i << " belongs to no element";
}

// =============================================================================
// Test 16 - Check that boundary groups are consistent with boundary nodes
// =============================================================================
TEST_F(StructuredMesh2DTest, BoundaryGroupsConsistentWithBoundaryNodes)
{
    for (const auto& [tag, nodeIDs] : mesh.getBoundaryGroups())
    {
        for (int nodeID : nodeIDs)
        {
            EXPECT_TRUE(mesh.isNodeBoundary(nodeID)) << "Node " << nodeID << " in group '" << tag << "' is not flagged as boundary";

            const BoundaryNode2D& bn = mesh.getBoundaryNode(nodeID);
            auto it = std::find(bn.tags_.begin(), bn.tags_.end(), tag);
            EXPECT_NE(it, bn.tags_.end())  << "Node " << nodeID << " missing tag '" << tag << "' in its BoundaryNode2D";
        }
    }
}