#include <gtest/gtest.h>

#include <unordered_set>
#include <vector>
#include "StructuredMesh2D.hpp"

// =============================================================================
// Fixture - Avoids repeated allocation
// =============================================================================
class StructuredMesh2DTest : public testing::Test
{
    protected: 
        static constexpr int nx = 11, ny = 21;
        static constexpr double left = 0, right = 1, bottom = 0, top = 1;
        std::vector<spatial::Node2D> nodes;
        const spatial::StructuredMesh2D mesh{left,right,bottom,top,nx,ny};
};

// =============================================================================
// Test 1 - Check constructor from invalid dimensions
// =============================================================================
TEST(StructuredMesh2D, InvalidDimensionsThrow) 
{
    EXPECT_THROW(spatial::StructuredMesh2D(0,1,0,-1,10,10), std::invalid_argument);
    EXPECT_THROW(spatial::StructuredMesh2D(0,-1,0,1,10,10), std::invalid_argument);
}

// =============================================================================
// Test 2 - Check constructor from invalid (negative) number of nodes
// =============================================================================
TEST(StructuredMesh2D, InvalidNumberOfNodesThrow) 
{
    EXPECT_THROW(spatial::StructuredMesh2D(0,1,0,1,-10,10), std::invalid_argument);
    EXPECT_THROW(spatial::StructuredMesh2D(0,1,0,1,10,-10), std::invalid_argument);
}

// =============================================================================
// Test 3 - Check constructor
// =============================================================================
TEST_F(StructuredMesh2DTest, ConstructsMeshFromDomainSides) 
{
    const spatial::StructuredMesh2D mesh(0,1,0,1,nx,ny);

    EXPECT_DOUBLE_EQ(mesh.getDx(), 1. / (nx - 1));
    EXPECT_DOUBLE_EQ(mesh.getDy(), 1. / (ny - 1));
}

// =============================================================================
// Test 4 - Check constructor using spatial::Domain struct
// =============================================================================
TEST_F(StructuredMesh2DTest, ConstructsMeshFromDomain) 
{
    const spatial::Domain2D domain{0, 1, 0, 1};
    const spatial::StructuredMesh2D mesh(domain,nx,ny);

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
    const std::vector<spatial::BoundaryNode2D>& t_corners = mesh.getBoundaryNodes();
    std::unordered_set<int> corners{0, 10, 220, 230};
    
    for (auto node : t_corners)
    {
        if (node.sides_.size() == 2)
        {
            EXPECT_TRUE(corners.count(node.nodeID_));
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
// Test 10 - Check correct size of sides in getBoundaries()
// =============================================================================
TEST_F(StructuredMesh2DTest, BoundaryNodesAssignation)
{
    const std::array<std::vector<int>, 4>& boundaries = mesh.getBoundaries();
    for (int i = 0; i < 4; ++i) 
    {
        if (i == sideToIndex(spatial::DomainSide::Left) || i == sideToIndex(spatial::DomainSide::Right)) EXPECT_EQ(boundaries[i].size(), ny);
        else EXPECT_EQ(boundaries[i].size(), nx);
    }
}

// =============================================================================
// Test 11 - Check that getInnerNodes() and getBoundaryNodes() return, 
//           respectively, only inner and boundary nodes 
// =============================================================================
TEST_F(StructuredMesh2DTest, InnerBoundaryNodesSeparation)
{
    for (int nodeID : mesh.getInnerNodes()) 
    {
        EXPECT_FALSE(mesh.isBoundary(nodeID));
    }

    for (auto node : mesh.getBoundaryNodes()) 
    {
        EXPECT_TRUE(mesh.isBoundary(node.nodeID_));
    }
}