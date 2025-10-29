// Parameters
nx = 20;
ny = 5;
u = -2;

xmin = -50;
xmax = 50;
yheight = 4;

UpperID[] = {};
LowerID[] = {};
LeftID[] = {};
RightID[] = {};

hp = 1;

// Horizontal edges
For i In {1:nx}
    x = xmin + (i-1)*(xmax - xmin)/(nx-1);

    L = 100;
    beta = 1.875/L;
    C = (Cosh(beta*L) + Cos(beta*L)) / (Sinh(beta*L) + Sin(beta*L));
    y_bottom  = Cosh(beta*(x+50)) - Cos(beta*(x+50)) - C * (Sinh(beta*(x+50)) - Sin(beta*(x+50)));
    y_top    = u * y_bottom + yheight;

    Point(hp) = {x, y_top, 0, 1.0};
    UpperID[i] = hp;
    hp = hp + 1;

    Point(hp) = {x, u * y_bottom, 0, 1.0};
    LowerID[i] = hp;
    hp = hp + 1;
EndFor

// Vertical edges
For i In {2:ny-1}
    y = (i-1) * yheight / (ny-1);

    L = 100;
    beta = 1.875/L;
    C = (Cosh(beta*L) + Cos(beta*L)) / (Sinh(beta*L) + Sin(beta*L));
    x = 100;
    y_bottom  = Cosh(beta*x) - Cos(beta*x) - C * (Sinh(beta*x) - Sin(beta*x));
    Point(hp) = {50, y + u * y_bottom, 0, 1.0};
    RightID[i] = hp;
    hp = hp + 1;

    L = 100;
    beta = 1.875/L;
    C = (Cosh(beta*L) + Cos(beta*L)) / (Sinh(beta*L) + Sin(beta*L));
    x = 0;
    y_bottom  = Cosh(beta*x) - Cos(beta*x) - C * (Sinh(beta*x) - Sin(beta*x));
    Point(hp) = {-50, y + u * y_bottom, 0, 1.0};
    LeftID[i] = hp;
    hp = hp + 1;
EndFor

// Create point lists
UpperPts[] = {}; LowerPts[] = {};
RightPts[] = {}; LeftPts[] = {};

For i In {1:nx}
  UpperPts[] += {UpperID[i]};
  LowerPts[] += {LowerID[i]};
EndFor

For i In {2:ny-1}
  RightPts[] += {RightID[i]};
  LeftPts[] += {LeftID[i]};
EndFor

// Build splines
Spline(1) = {UpperPts[]};
Spline(2) = {LowerPts[]};
Spline(3) = {LowerPts[nx-1], RightPts[], UpperPts[nx-1]};
Spline(4) = {LowerPts[0], LeftPts[], UpperPts[0]};

// Correct orientation
Curve Loop(1) = {2, 3, -1, -4};
Plane Surface(1) = {1};

Mesh 2;




