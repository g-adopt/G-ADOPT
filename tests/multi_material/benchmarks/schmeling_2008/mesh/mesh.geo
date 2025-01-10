horizontal_resolution = 15e3;

domain_dimension_x = 3e6;
domain_dimension_y = 7.5e5;

layer_min_y = 6.5e5;
layer_thickness = 8e4;
layer_vertical_resolution = 5e3;

Point(1) = {0, 0, 0, horizontal_resolution};
Point(2) = {domain_dimension_x, 0, 0, horizontal_resolution};

Line(1) = {1, 2};

Extrude {0, layer_min_y, 0} {
  Curve{1}; Layers {25}; Recombine;  // Vertical resolution: 26 km
}
Extrude {0, layer_thickness, 0} {
  Curve{2}; Layers {layer_thickness / layer_vertical_resolution}; Recombine;
}
Extrude {0, domain_dimension_y - layer_min_y - layer_thickness, 0} {
  Curve{6}; Layers {4}; Recombine;  // Vertical resolution: 5 km
}

Physical Curve(1) = {3, 7, 11};
Physical Curve(2) = {4, 8, 12};
Physical Curve(3) = {1};
Physical Curve(4) = {10};

Physical Surface(1) = {5, 9, 13};
