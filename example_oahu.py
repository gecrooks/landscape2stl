#!/usr/bin/env python

from landscape2stl import create_stl, STLParameters


params = STLParameters(
    scale=250_000,
    magnet_spacing=0.05,
    exaggeration=2,
    magnet_holes=True,
    bolt_holes=False,
    pin_holes=False,
    )

ee = -158.3
nn = 21.30
delta = 0.1

# Oahu 
for e in range(0, 7):
    for n in range(0, 6):
        name = f"oahu_250000_{e}{n}"
        north = nn + n * delta
        east = ee + e * delta
        print(name, north, east, north-delta, east+delta)
        create_stl(params, (north, east, north-delta, east+delta), name=name, verbose=True)
