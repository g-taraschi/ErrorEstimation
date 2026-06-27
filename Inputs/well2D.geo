SetFactory("OpenCASCADE"); // Geometry kernel

Geometry.MatchMeshTolerance = 1e-08;
Geometry.Tolerance = 1e-10;
Mesh.ToleranceReferenceElement = 1e-10;
General.Terminal = 1;
Geometry.SnapPoints = 1;

Lw = 100;
Lr = 200;
Wr = 100;
Hr = 10;

// External points
p1 = newp; Point(p1) = {0.0, 0.0, 0.0, 1.0};
p2 = newp; Point(p2) = {1.0, 0.0, 0.0, 1.0};
p3 = newp; Point(p3) = {1.0, 1.0, 0.0, 1.0};
p4 = newp; Point(p4) = {0.0, 1.0, 0.0, 1.0};

// Internal points
p5 = newp; Point(p5) = {0.35, 0.35, 0.0, 1.0};
p6 = newp; Point(p6) = {0.65, 0.35, 0.0, 1.0};
p7 = newp; Point(p7) = {0.65, 0.65, 0.0, 1.0};
p8 = newp; Point(p8) = {0.35, 0.65, 0.0, 1.0};

// Create outer square (BigBox)
l1 = newl; Line(l1) = {p1, p2};
l2 = newl; Line(l2) = {p2, p3};
l3 = newl; Line(l3) = {p3, p4};
l4 = newl; Line(l4) = {p4, p1};

// Create inner square (SmallBox)
l5 = newl; Line(l5) = {p5, p6};
l6 = newl; Line(l6) = {p6, p7};
l7 = newl; Line(l7) = {p7, p8};
l8 = newl; Line(l8) = {p8, p5};

// Diagonal lines
l9 = newl; Line(l9) = {p1, p5};
l10 = newl; Line(l10) = {p2, p6};
l11 = newl; Line(l11) = {p3, p7};
l12 = newl; Line(l12) = {p4, p8};

cl1 = newcl; Curve Loop(cl1) = {l1, l10, -l5, -l9};
cl2 = newcl; Curve Loop(cl2) = {l2, l11, -l6, -l10};
cl3 = newcl; Curve Loop(cl3) = {l3, l12, -l7, -l11};
cl4 = newcl; Curve Loop(cl4) = {l4, l9, -l8, -l12};

sf1 = news; Plane Surface(sf1) = {cl1};
sf2 = news; Plane Surface(sf2) = {cl2};
sf3 = news; Plane Surface(sf3) = {cl3};
sf4 = news; Plane Surface(sf4) = {cl4};

Transfinite Line {l1, l3, l5, l7} = 5;
Transfinite Line {l2, l4, l6, l8} = 5;
Transfinite Line {l9, l10, l11, l12} = 5;

Transfinite Surface {sf1, sf2, sf3, sf4};

Physical Surface("Domain") = {sf1, sf2, sf3, sf4};
Physical Curve("Inner") = {l5, l6, l7, l8};
Physical Curve("Outer") = {l1, l2, l3, l4};

Recombine Surface "*";
Mesh 2;
Save "well2D.msh";
