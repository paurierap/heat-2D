#ifndef UNSTRUCTUREDMESH_HPP
#define UNSTRUCTUREDMESH_HPP

#ifndef HEAT2D_HAS_GMSH
#error "UnstructuredMesh2D requires Gmsh. Configure with HEAT2D_ENABLE_GMSH=ON."
#endif

#include <string>

#include "Mesh2D.hpp"

namespace heat2d::mesh
{

class UnstructuredMesh2D : public Mesh2D
{
    private:
        double meshSize_; 

    protected:
        void meshDomain() override;

    public:

        // Constructors
        UnstructuredMesh2D(const std::string& gmshFile);
        
        // Getters
        inline double getMeshSize() const override {return meshSize_;};

        // Specific helpers
        double getElementArea(std::size_t elementID) const override;
};

};// namespace

#endif // ifndef UNSTRUCTUREDMESH_HPP