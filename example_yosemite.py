#!/usr/bin/env python

from landscape2stl import create_stl, STLParameters, BBox, yosemite_quadrangles, presets


params = STLParameters(
    scale=62_500,
    exaggeration=1,
    magnet_holes=True,
    bolt_holes=False,
    pin_holes=True,
    )


for name in yosemite_quadrangles.keys():
    coords, _ = presets["quad_"+name] 
    create_stl(params, coords,  name, verbose=True)
