nx = 20;
ny = 5;
L = 100;

// Define number of terms explicitly
n = 2;

// Define arrays (using parentheses!)
coeff(1) = -2;
coeff(2) = 1;

beta(1) = 1.875 / L;
beta(2) = 4.694 / L;

// Initialize
y_bottom = 0;
x = 0;

// Loop over terms (use explicit range 1:2)
For j In {1:n}
    C = (Cosh(beta(j)*L) + Cos(beta(j)*L)) / (Sinh(beta(j)*L) + Sin(beta(j)*L));
    y_bottom = y_bottom + coeff(j) * (Cosh(beta(j)*(x+50)) - Cos(beta(j)*(x+50)) - C * (Sinh(beta(j)*(x+50)) - Sin(beta(j)*(x+50))));
EndFor

