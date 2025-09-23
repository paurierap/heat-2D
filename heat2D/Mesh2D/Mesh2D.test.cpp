#include <gtest/gtest.h>
#include <vector>
#include "StructuredMesh2D.hpp"

class StructuredMesh2DTest : public testing::Test
{
    protected: 
        const int nx = 11, ny = 11;
        std::vector<mesh::Node2D> nodes;

        void SetUp() override 
        {
            nodes.resize(nx * ny);
            for (int row = 0; row < nx; ++row)
            {
                for (int col = 0; col < ny; ++col)
                {
                    int nodeID = nx * row + col;
                    nodes[nodeID] = mesh::Node2D{nodeID, col * 0.1, row * 0.1};
                }
            }
        };
};

TEST_F(StructuredMesh2DTest, ConstructsMeshFromSides) 
{
    mesh::StructuredMesh2D mesh(0,1,0,1,nx,ny);

    EXPECT_TRUE(std::equal(nodes.begin(), nodes.end(), mesh.getMesh().begin(), 
    [](const mesh::Node2D& a, const mesh::Node2D& b) {
        return a.nodeID_ == b.nodeID_ && a.x_ == b.x_ && a.y_ == b.y_;
    }));

}

TEST_F(StructuredMesh2DTest, ConstructsMeshFromDomain) 
{
    const mesh::Domain2D domain{0, 1, 0, 1};
    mesh::StructuredMesh2D mesh(domain,nx,ny);

    EXPECT_TRUE(std::equal(nodes.begin(), nodes.end(), mesh.getMesh().begin(), 
    [](const mesh::Node2D& a, const mesh::Node2D& b) {
        return a.nodeID_ == b.nodeID_ && a.x_ == b.x_ && a.y_ == b.y_;
    }));
}

TEST(StructuredMesh2D, InvalidDimensionsThrow) 
{
    EXPECT_THROW(mesh::StructuredMesh2D(0,1,0,-1,10,10), std::invalid_argument);
    EXPECT_THROW(mesh::StructuredMesh2D(0,-1,0,1,10,10), std::invalid_argument);
}

TEST(StructuredMesh2D, InvalidNumberOfNodesThrow) 
{
    EXPECT_THROW(mesh::StructuredMesh2D(0,1,0,1,-10,10), std::invalid_argument);
    EXPECT_THROW(mesh::StructuredMesh2D(0,1,0,1,10,-10), std::invalid_argument);
}