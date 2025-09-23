#ifndef MESH_HPP
#define MESH_HPP

#include <vector>
#include <stdexcept>

namespace mesh
{

struct Node2D
{
    int nodeID_;
    double x_;
    double y_;
};

struct Domain2D
{
    double left_;
    double right_;
    double bottom_;
    double top_;   
};

class Mesh2D
{
    protected:
        std::vector<Node2D> nodes_;
        Domain2D domain_;

        virtual void meshDomain() = 0;

    public:
        Mesh2D(double left, double right, double bottom, double top) 
        {
            if (left > right || bottom > top) throw std::invalid_argument("Inconsistent geometrical constraints. The left (or bottom) side cannot be larger than the right (or top) side.");
            domain_ = {left, right, bottom, top};
        };

        Mesh2D(const Domain2D& domain) 
        : Mesh2D(domain.left_, domain.right_, domain.bottom_, domain.top_) {};

        Mesh2D(const Mesh2D&) = delete;
        Mesh2D& operator=(const Mesh2D&) = delete;
        Mesh2D(const Mesh2D&&) = delete;
        Mesh2D& operator=(const Mesh2D&&) = delete;

        virtual ~Mesh2D() {};

        const std::vector<Node2D>& getMesh() const
        {
            return nodes_;
        }
};

} // namespace
#endif // ifndef Mesh_HPP