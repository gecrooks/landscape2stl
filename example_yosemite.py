#!/usr/bin/env python

from landscape2stl import create_stl, STLParameters


params = STLParameters(
    scale=62_500,
    magnet_spacing=0.025,
    exaggeration=1,
    magnet_holes=True,
    bolt_holes=False,
    pin_holes=False,
    pitch=0.2,
    )

ee = -119.80
nn = 37.70
e_delta = 0.15
n_delta = 0.1


# Oahu 
for e in range(0, 5):
    for n in range(0, 3):
        name = f"yosemite_62500_{e}{n}"
        north = nn + n * n_delta
        east = ee + e * e_delta
        print(name, north, east, north-n_delta, east+e_delta)
        create_stl(params, (north, east, north-n_delta, east+e_delta), name=name, verbose=True)
