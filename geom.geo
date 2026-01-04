// ----------------------------------------------------------------------------------
//
//  Gmsh GEO generation
//
//  Generating the half geometry based on the symmetry axis of the problem definiton
//  consisting of a wire at hight 10R from the base of an open domain. The domain has
//  40R semiwidth and 100R height based on the problem description
//
// ----------------------------------------------------------------------------------


// Setting the appropiate kernel
SetFactory("OpenCASCADE");

// Setting up the parameters for the geometry to be modelled for multiple 
// experimental setups

R_placeholder = 1;
resolution_placeholder = 50;

R = R_placeholder;
w = 40 * R;
h = 100 * R;
lc1 = R_placeholder / 10;
lc2 = R_placeholder / 1000;
resolution = resolution_placeholder;

// Setting the points of the domain
Point(1) = {0.0, 0.0,    0.0, lc1};
Point(2) = {w,   0.0,    0.0, lc1};
Point(3) = {w,   h,      0.0, lc1};
Point(4) = {0.0, h,      0.0, lc1};
Point(5) = {0.0, 10 * R, 0.0, lc2};
Point(6) = {0.0, 11 * R, 0.0, lc2};
Point(7) = {0.0, 12 * R, 0.0, lc2};

// Setting the domain edges
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 7};
Line(5) = {5, 1};
Line(6) = {7, 5};

Circle(7) = {7, 6, 5};

// Setting the outside curve
Curve Loop(1) = {1, 2, 3, 4, 7, 5};
Curve Loop(2) = {7, -6};
//Curve (3) = {6};
//Curve (4) = {7};

// Setting the surface
Plane Surface(1) = {1};
Plane Surface(2) = {2};

// Setting Physical surfaces
Physical Curve(100) = {4,5};            // symmetry boundary of the air domain
Physical Curve(101) = {1,2,3};          // outer boundary of the air domain
Physical Curve(102) = {7};              // outer boundary of the wire domain
Physical Curve(103) = {6};              // symmetry boundary of the wire domain

Physical Surface(10) = {2};             // wire domain
Physical Surface(11) = {1};             // air domain

// Setting mesh resolution on the wire
Transfinite Line{7} = resolution;
Transfinite Line{6} = resolution / 10;
Transfinite Line{1} = resolution ;
Transfinite Line{2} = resolution * 4;
Transfinite Line{3} = resolution ;
Transfinite Line{4} = resolution * 3;
Transfinite Line{5} = resolution / 2;

//Setting mesh on air
//Transfinite Surface{1} = {1, 2, 3, 4};


// Meshing the geometry
Mesh 2;

// Finally we apply an elliptic smoother to the grid to have a more regular
// mesh:
Mesh.Smoothing = 100;

// Saving the geometry
Save "plume.msh";

Exit;
