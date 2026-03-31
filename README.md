# Heat2D

A lightweight numerical library for solving 2D parabolic and elliptic PDEs in C++.

This project is designed as a clean foundation for:
- structured mesh generation in 2D
- Dirichlet and Neumann boundary conditions
- finite difference spatial discretization
- convergence verification via unit tests

## Features

- Structured 2D meshes with easy node/coordinate access
- Boundary condition abstraction (Dirichlet & Neumann)
- Second-order finite difference discretization for Laplace/Poisson
- Strong test suite validating:
  - stencil correctness
  - operator consistency
  - convergence rates (2nd order)
- Googletest-based infrastructure with CMake integration

## Build & Test

```bash
mkdir build && cd build
cmake ..
make
ctest --verbose
```

## Notes


## TO DO:

1. UnstructuredMesh2D and the implementation of FEM.
2. Study change from reference to mesh in SpatialDiscretization to using a shared_ptr.