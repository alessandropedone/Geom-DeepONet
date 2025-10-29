// Parameters
nx = 20;
ny = 5;

L = 100;
n = 4;

coeff(1) = 2;
beta(1) = 0.596864 * 3.141666/L;

coeff(2) = 0.3;
beta(2) = 1.49418 * 3.141666 / L;

coeff(3) = 0.2;
beta(3) = 2.50025 * 3.141666 / L;

coeff(4) = 0.1;
beta(4) = 3.49999 * 3.141666 / L;

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

    y_bottom = 0;
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

// Vertical edges
For i In {2:ny-1}

    y = (i-1) * yheight / (ny-1);

    y_bottom = 0;
    x = 50;
    For j In {1:n}
        C = (Cosh(beta(j)*L) + Cos(beta(j)*L)) / (Sinh(beta(j)*L) + Sin(beta(j)*L));
        y_bottom  = y_bottom + coeff(j) * (Cosh(beta(j)*(x+50)) - Cos(beta(j)*(x+50)) - C * (Sinh(beta(j)*(x+50)) - Sin(beta(j)*(x+50))));
    EndFor

    Point(hp) = {50, y + y_bottom, 0, 1.0};
    RightID[i] = hp;
    hp = hp + 1;

    y_bottom = 0;
    x = -50;
    For j In {1:n}
        C = (Cosh(beta(j)*L) + Cos(beta(j)*L)) / (Sinh(beta(j)*L) + Sin(beta(j)*L));
        y_bottom  = y_bottom + coeff(j) * (Cosh(beta(j)*(x+50)) - Cos(beta(j)*(x+50)) - C * (Sinh(beta(j)*(x+50)) - Sin(beta(j)*(x+50))));
    EndFor

    Point(hp) = {-50, y + y_bottom, 0, 1.0};
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




