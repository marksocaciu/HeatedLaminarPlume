// ----------------------------------------------------------------------------------
//
//  Gmsh GEO generation
//
//  Full geometry obtained by mirroring the former half-domain about x = 0.
//  Domain width = 80R on each side, total width = 160R, height = 100R.
//
// ----------------------------------------------------------------------------------

SetFactory("OpenCASCADE");
General.NumThreads = 12;

// Parameters
R_placeholder = 1;
resolution_placeholder = 100;

R = R_placeholder;
w = 40 * R;                  // half-width
h = 100 * R;
lc_far  = 5e-4;
lc_axis = 1e-6;
lc_wire = 1e-7;
resolution = resolution_placeholder;
dist = R * 50;

// -----------------------------------------------------------------------------
// Points
// -----------------------------------------------------------------------------

// Outer rectangle
Point(1) = {0.0, 0.0, 0.0, lc_far};      // bottom symmetry point
Point(2) = { w, 0.0, 0.0, lc_far};
Point(3) = { w,   h, 0.0, lc_far};
Point(4) = {0.0, h, 0.0, lc_far};        // upper symmetry point
Point(5) = {-w,   h, 0.0, lc_far};
Point(6) = {-w, 0.0, 0.0, lc_far};

// Symmetry axis points wire
Point(7)  = {0.0,  h/10 + 10*R, 0.0, lc_wire};   // lower point of upper semicircle
Point(8)  = {0.0,  h/10 + 11*R, 0.0, lc_wire};   // center of upper semicircle
Point(9)  = {0.0,  h/10 + 12*R, 0.0, lc_wire};   // top point of upper semicircle

// Symmetry axis points refinement
Point(10)  = {0.0,  h/10 - 12*R, 0.0, lc_axis};   // lower point of lower semicircle
Point(11) = {0.0,  h/10 - 12*R + dist, 0.0, lc_axis};   // center of lower semicircle
Point(12)  = {0.0,  h/10 - 12*R + 2*dist, 0.0, lc_axis}; // top point of lower semicircle


// -----------------------------------------------------------------------------
// Outer boundary
// -----------------------------------------------------------------------------
Line(1) = {1, 2};   // bottom-right
Line(2) = {2, 3};   // right
Line(3) = {4, 3};   // top-right
Line(4) = {4, 5};   // top-left
Line(5) = {5, 6};   // left
Line(6) = {1, 6};   // bottom-left

// -----------------------------------------------------------------------------
// Internal wire
// -----------------------------------------------------------------------------
Circle(7)  = {0.0,  h/10 + 11*R, 0.0, R};    // right semicircle
//Circle(8)  = {7, 8, 9};    // left semicircle

//-----------------------------------------------------------------------------
// Internal refinement
// -----------------------------------------------------------------------------
Circle(9) = {0.0,  h/10 - 12*R + dist, 0.0, dist};   // right semicircle
//Circle(10) = {10, 11, 12};   // left semicircle

// Centerline
Line(8) = {9, 12};      // wire - refinement internal split
// Line(10) = {12, 4};      // refinement - boundary internal split
// Line(9) = {7, 5};      // wire symmetry line
// Line(11)    = {7, 9};       // internal vertical connector
// Line(12)   = {5, 8};       // internal vertical connector

// -----------------------------------------------------------------------------
// Surface loops
// -----------------------------------------------------------------------------

// Full outer air region cut by internal line-source geometry
Curve Loop(1) = {1, 2, -3, 4, 5, -6};

// Wire domain
Curve Loop(2) = {7};

// Central air strip around wire
Curve Loop(3) = {9};

// Surfaces
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

// Make the topology conformal: split all overlapping surfaces
//BooleanFragments{ Surface{1}; Delete; }{ Surface{2}; Surface{3}; Curve{8}; Curve{10}; Delete; }
BooleanFragments{ Surface{1}; Delete; }{ Surface{2}; Surface{3}; Curve{8}; Delete; }

// -----------------------------------------------------------------------------
// Physical groups
// -----------------------------------------------------------------------------

// No symmetry boundary anymore
//Physical Curve(101) = {20, 19, 18, 17, 16, 15};   // outer boundary of air domain
Physical Curve(101) = {9, 10, 11, 13, 14, 12};   // outer boundary of air domain

// Wire-air interface
Physical Curve(102) = {17, 18};            // outer boundary of wire domain
Physical Curve(103) = {11, 12};

Physical Surface(10) = {2};           // wire domain
Physical Surface(11) = {4, 3};        // air domain

// -----------------------------------------------------------------------------
// Mesh resolution
// -----------------------------------------------------------------------------
// Wire
Transfinite Line{18}  = resolution * 2.5 * 1.5;
Transfinite Line{17}  = resolution * 2.5 * 0.5;

// Outer
Transfinite Line{10}  = resolution * 0.2;
Transfinite Line{11}  = resolution * 0.3;
Transfinite Line{13}  = resolution * 0.2;
Transfinite Line{14}  = resolution * 0.2;
Transfinite Line{12}  = resolution * 0.3;
Transfinite Line{9}  = resolution * 0.2;

// Refinement
Transfinite Line{16}  = resolution * 3 * 0.25;
Transfinite Line{15}  = resolution * 3 * 0.75;

// Centerline
Transfinite Line{8}  = resolution * 3 Using Progression 1.006;
//Transfinite Line{10}  = resolution * 8 Using Progression 1.003;

// Mesh
Mesh 2;

Mesh.Smoothing = 100;

Save "plume.msh";
Exit;
