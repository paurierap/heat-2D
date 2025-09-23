#ifndef STRUCTUREDMESH_HPP
#define STRUCTUREDMESH_HPP

#include "Mesh2D.hpp"

namespace mesh 
{

class StructuredMesh2D : public Mesh2D
{
    private:
        int nx_;
        int ny_;

    protected:
        void meshDomain() override
        {
            double dx = (domain_.right_ - domain_.left_) / (nx_ - 1);
            double dy = (domain_.top_ - domain_.bottom_) / (ny_ - 1);
            nodes_.reserve(nx_ * ny_);

            for (int row = 0; row < ny_; ++row)
            {
                for (int col = 0; col < nx_; ++col)
                {
                    int nodeID = nx_ * row + col;
                    nodes_[nodeID] = Node2D{nodeID, domain_.left_ + col * dx, domain_.bottom_ + row * dy};
                }
            }
        };

    public:
        StructuredMesh2D(double left, double right, double bottom, double top, int nx, int ny) 
        : Mesh2D(left, right, bottom, top)
        {
            if (nx <= 0 || ny <= 0) throw std::invalid_argument("Number of nodes must be positive.");
            nx_ = nx;
            ny_ = ny;

            meshDomain();
        };

        StructuredMesh2D(const Domain2D& domain, int nx, int ny) 
        : StructuredMesh2D(domain.left_, domain.right_, domain.bottom_, domain.top_, nx, ny) {};
};

} // namespace 

#endif // ifndef UnstructuredMesh_HPP
