// S shape, mirrored, with physical tags for BCs
lc = 0.01;

Point(1) = {0, 1.3, 0, lc};
Point(2) = {0, 1.1, 0, lc};
Point(3) = {0, 0.1, 0, lc};
Point(4) = {0, -0.1, 0, lc};
Point(5) = {0, -1.1, 0, lc};
Point(6) = {0, -1.3, 0, lc};
Point(7) = {0, 0.6, 0, lc};
Point(8) = {0, -0.6, 0, lc};

Point(9)  = {-0.7, 0.6, 0, lc};
Point(10) = {-0.5, 0.6, 0, lc};
Point(11) = {0.7, -0.6, 0, lc};
Point(12) = {0.5, -0.6, 0, lc};

Line(1) = {1, 2};
Line(2) = {5, 6};

Circle(3)  = {1, 7, 9};
Circle(4)  = {9, 7, 4};
Circle(5)  = {2, 7, 10};
Circle(6)  = {10, 7, 3};
Circle(7)  = {3, 8, 11};
Circle(8)  = {11, 8, 6};
Circle(9)  = {4, 8, 12};
Circle(10) = {12, 8, 5};

Curve Loop(1) = {1, 5, 6, 7, 8, -2, -10, -9, -4, -3};
Plane Surface(1) = {1};

// --- Physical tags ---
Physical Curve("hot_end")    = {1};
Physical Curve("cold_end")   = {2};
Physical Curve("insulated_top_hook")    = {3, 4, 5, 6};
Physical Curve("insulated_bottom_hook") = {7, 8, 9, 10};

Physical Surface("domain") = {1};
