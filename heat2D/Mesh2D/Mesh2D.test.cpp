#include <gtest/gtest.h>

#include <unordered_set>
#include <vector>
#include "StructuredMesh2D.hpp"

class StructuredMesh2DTest : public testing::Test
{
    protected: 
        const int nx = 11, ny = 21;
        const double left = 0, right = 1, bottom = 0, top = 1;
        std::vector<spatial::Node2D> nodes;
        spatial::StructuredMesh2D mesh{left,right,bottom,top,nx,ny};
};

TEST(StructuredMesh2D, InvalidDimensionsThrow) 
{
    EXPECT_THROW(spatial::StructuredMesh2D(0,1,0,-1,10,10), std::invalid_argument);
    EXPECT_THROW(spatial::StructuredMesh2D(0,-1,0,1,10,10), std::invalid_argument);
}

TEST(StructuredMesh2D, InvalidNumberOfNodesThrow) 
{
    EXPECT_THROW(spatial::StructuredMesh2D(0,1,0,1,-10,10), std::invalid_argument);
    EXPECT_THROW(spatial::StructuredMesh2D(0,1,0,1,10,-10), std::invalid_argument);
}

TEST_F(StructuredMesh2DTest, ConstructsMeshFromDomainSides) 
{
    spatial::StructuredMesh2D mesh(0,1,0,1,nx,ny);

    EXPECT_DOUBLE_EQ(mesh.getDx(), 1. / (nx - 1));
    EXPECT_DOUBLE_EQ(mesh.getDy(), 1. / (ny - 1));
}

TEST_F(StructuredMesh2DTest, ConstructsMeshFromDomain) 
{
    const spatial::Domain2D domain{0, 1, 0, 1};
    spatial::StructuredMesh2D mesh(domain,nx,ny);

    EXPECT_DOUBLE_EQ(mesh.getDx(), (domain.right_ - domain.left_) / (nx - 1));
    EXPECT_DOUBLE_EQ(mesh.getDy(), (domain.top_ - domain.bottom_) / (ny - 1));
}

TEST_F(StructuredMesh2DTest, DxDyCalculation) 
{
    EXPECT_DOUBLE_EQ(mesh.getDx(), (mesh.getDomain().right_ - mesh.getDomain().left_) / (nx - 1));
    EXPECT_DOUBLE_EQ(mesh.getDy(), (mesh.getDomain().top_ - mesh.getDomain().bottom_) / (ny - 1));
}

TEST_F(StructuredMesh2DTest, NodeCounting)
{
    EXPECT_EQ(mesh.getNx(), nx);
    EXPECT_EQ(mesh.getNy(), ny);
    EXPECT_EQ(mesh.getNodes().size(), mesh.getNx() * mesh.getNy());
    EXPECT_EQ(mesh.getBoundaryNodes().size(), 2 * (nx + ny) - 4);
    EXPECT_EQ(mesh.getBoundaryNodes().size() + mesh.getInnerNodes().size(), mesh.getNodes().size());
    EXPECT_EQ(mesh.getNodeID(6, 12), 12 * nx + 6);
}

TEST_F(StructuredMesh2DTest, NodeOutOfBounds)
{
    EXPECT_FALSE(mesh.getNodeID(-1, 1));
    EXPECT_FALSE(mesh.getNodeID(1, -1));
    EXPECT_FALSE(mesh.getNodeID(nx, 1));
    EXPECT_FALSE(mesh.getNodeID(1, ny));
}

TEST_F(StructuredMesh2DTest, CheckCorners)
{
    std::vector<spatial::BoundaryNode2D> t_corners = mesh.getBoundaryNodes();
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

TEST_F(StructuredMesh2DTest, MeshGeneration)
{
    double dx = mesh.getDx(), dy = mesh.getDy();

    for (int row = 0; row < mesh.getNy(); ++row) 
    {
        for (int col = 0; col < mesh.getNx(); ++col) 
        {
            int nodeID = row * mesh.getNx() + col;
            EXPECT_NEAR(mesh.getNodes()[nodeID].x_, mesh.getDomain().left_ + col*dx, 1e-12);
            EXPECT_NEAR(mesh.getNodes()[nodeID].y_, mesh.getDomain().bottom_ + row*dy, 1e-12);
        }
    }
}

TEST_F(StructuredMesh2DTest, BoundaryNodesAssignation)
{
    for (const auto& [side, nodes] : mesh.getBoundaries()) 
    {
        if (side == spatial::DomainSide::Left || side == spatial::DomainSide::Right) EXPECT_EQ(nodes.size(), ny);
        else EXPECT_EQ(nodes.size(), nx);
    }
}

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