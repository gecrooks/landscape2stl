#!/usr/bin/env python

from landscape2stl import create_stl, STLParameters, BBox


params = STLParameters(
    scale=62_500,
    magnet_spacing=0.025,
    exaggeration=1,
    magnet_holes=True,
    bolt_holes=True,
    pin_holes=True,
    pitch=0.2,
    )

yosemite_boundary: BBox = (37.60, -119.80, 37.90, -119.35)  # south, west, north, east
lat_delta = 0.1
long_delta = 0.15


south, west, north, east = yosemite_boundary

slab_west = west
while slab_west < east:
    slab_south = south
    while slab_south < north:
        create_stl(params, (slab_south, slab_west, slab_south+lat_delta, slab_west+long_delta),  verbose=True)
        slab_south += lat_delta
    slab_west += long_delta


