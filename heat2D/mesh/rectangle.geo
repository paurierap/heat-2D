// Define mesh element size
lc = 0.1;

// Define Rectangle (X, Y, Z, dx, dy)
Rectangle(1) = {0.0, 0.0, 0.0, 2.0, 1.0, lc};

// Generate 2D mesh
Mesh 2;

// Save to file
Save "rectangle.msh";

