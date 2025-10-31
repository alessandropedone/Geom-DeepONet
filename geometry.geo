// Set the geometry kernel
SetFactory("OpenCASCADE");

//---------------------------------------
// Rectangle 1 (top)
//---------------------------------------

// Parameters for the deformation
nx = 50;

L = 100;
n = 4;

coeff(1) = 2;
beta(1) = 0.596864 * 3.141666 / L;

coeff(2) = 0.3;
beta(2) = 1.49418 * 3.141666 / L;

coeff(3) = 0.2;
beta(3) = 2.50025 * 3.141666 / L;

coeff(4) = 0.1;
beta(4) = 3.49999 * 3.141666 / L;


xmin = -50;
xmax = 50;
yheight = 4;

// Indexes of the points
UpperID[] = {};
LowerID[] = {};
LeftID[] = {};
RightID[] = {};
hp = 1;

// Horizontal edges
For i In {1:nx}
    x = xmin + (i-1)*(xmax - xmin)/(nx-1);

    y_bottom = 1;
    For j In {1:n}
        C = (Cosh(beta(j)*L) + Cos(beta(j)*L)) / (Sinh(beta(j)*L) + Sin(beta(j)*L));
        y_bottom  = y_bottom + coeff(j) * (Cosh(beta(j)*(x+50)) - Cos(beta(j)*(x+50)) - C * (Sinh(beta(j)*(x+50)) - Sin(beta(j)*(x+50))));
    EndFor

    y_top = y_bottom + yheight;

    Point(hp) = {x, y_top, 0, 1.0};
    UpperID[i] = hp;
    hp = hp + 1;

    Point(hp) = {x, y_bottom, 0, 1.0};
    LowerID[i] = hp;
    hp = hp + 1;
EndFor

// Create point lists
UpperPts[] = {}; LowerPts[] = {};
For i In {1:nx}
  UpperPts[] += {UpperID[i]};
  LowerPts[] += {LowerID[i]};
EndFor

// Build splines
Spline(1) = {UpperPts[]};
Spline(2) = {LowerPts[]};
Line(3) = {LowerPts[nx-1], UpperPts[nx-1]};
Line(4) = {LowerPts[0], UpperPts[0]};

// 
Curve Loop(1) = {2, 3, -1, -4};
Plane Surface(1) = {1};



//---------------------------------------
// Rectangle 2 (bottom)
//---------------------------------------

Point(1001) = {-50, -1, 0, 1.0};
Point(1002) = { 50, -1, 0, 1.0};
Point(1003) = { 50,  -5, 0, 1.0};
Point(1004) = {-50,  -5, 0, 1.0};

Line(5) = {1001, 1002};
Line(6) = {1002, 1003};
Line(7) = {1003, 1004};
Line(8) = {1004, 1001};

Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2}; 


//---------------------------------------
// Circle
//---------------------------------------

// Define points
Point(1005) = {0, 0, 0};        // Center
Point(1006) = {200, 0, 0};     // Start point
Point(1007) = {0, 200, 0};     // 90 degrees
Point(1008) = {-200, 0, 0};    // 180 degrees
Point(1009) = {0, -200, 0};    // 270 degrees

// Define circle arcs (each needs start, center, end)
Circle(9) = {1006, 1005, 1007};
Circle(10) = {1007, 1005, 1008};
Circle(11) = {1008, 1005, 1009};
Circle(12) = {1009, 1005, 1006};

// Define curve loop and surface if needed
Curve Loop(3) = {9, 10, 11, 12};
Surface(3) = {3};

// Subtract Rectangles from Circle
BooleanDifference{ Surface{3}; Delete; }{ Surface{1}; Surface{2}; Delete; }

//---------------------------------------
// Transfinite Lines and Surface
//---------------------------------------
r = 7;
Transfinite Curve {1, 2} = 50*r Using Progression 1;
Transfinite Curve {3, 4} = 2*r Using Progression 1;
Transfinite Line {5, 7} = 50*r Using Progression 1;
Transfinite Line {6, 8} = 2*r Using Progression 1;
Transfinite Line {9, 10, 11, 12} = 20 Using Progression 1;

//---------------------------------------
// Define Physical Groups
//---------------------------------------                

//--- Physical Curves
// Physical groups for boundaries
Physical Line("force_segment", 10) = {2};
Physical Line("upper_plate", 11) = {1, 3, 4};
Physical Line("lower_plate", 12) = {5, 6, 7, 8};
Physical Line("boundary", 20) = {9, 10, 11, 12};

//--- Physical Surfaces
Physical Surface("space", 30) = {3};

//---------------------------------------
// 6. Generate the Mesh
//---------------------------------------
Mesh 2;  // 2D mesh generation