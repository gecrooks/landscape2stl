#!/usr/bin/env python

# Copyright 2023, Gavin E. Crooks and contributors
#
# This source code is licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
# This is a test for press-fit alignment magnets.
#
# Magnets (bought from amazon) are nominally 6mm x 2mm, but actual size should be checked with calipers.
# The size varies brand to brand and even batch to batch.
#
# Run script to create proto_magnets.stl, and print two copies.
#
# Press fit magnets into both (in opposite orientations). The trick to press fitting these magnets is to
# take a stack of 10 or more, hold over the opening, and give the top of the stack a tap with a
# mallet. If magnet fails to hold, a small drop of superglue can be added.
#
# Gavin E. Crooks 2023


import math
from math import pi

import numpy as np
from stl import mesh

magnet_diameter = 5.95  # magnet diameter in mm (measured with calipers)
magnet_depth = 1.65  # magnet depth in mm (measured with calipers)
magnet_padding = 0.25


diameter = magnet_diameter + magnet_padding * 2
depth = magnet_depth + magnet_padding
sides = 8  # Magnet hole is octagonal prism. Seems to work fine.


def PointsInCircum(r, n=100, z=0.0):
    return [
        (math.cos(2 * pi / n * x) * r, math.sin(2 * pi / n * x) * r, z)
        for x in range(0, n + 1)
    ]


circle = np.asarray(PointsInCircum(diameter / 2, sides, 0.0))

bot_circle = np.asarray(PointsInCircum(diameter / 2, sides, -depth))

center_bot = (0.0, 0.0, -depth)

triangles = []

hedge = 5

top_square = [
    [hedge, 0, 0],
    [hedge, hedge, 0],
    [0, hedge, 0],
    [-hedge, hedge, 0],
    [-hedge, 0, 0],
    [-hedge, -hedge, 0],
    [0, -hedge, 0],
    [hedge, -hedge, 0],
    [hedge, 0, 0],
]


for x in range(sides):
    tri = (circle[x], circle[x + 1], bot_circle[x])
    triangles.append(tri)
    tri = (bot_circle[x + 1], bot_circle[x], circle[x + 1])
    triangles.append(tri)
    tri = (bot_circle[x], bot_circle[x + 1], center_bot)
    triangles.append(tri)

    tri = (
        circle[x],
        top_square[x],
        circle[x + 1],
    )
    triangles.append(tri)
    tri = (top_square[x], top_square[x + 1], circle[x + 1])
    triangles.append(tri)


tnw = [hedge, hedge, 0]
tne = [hedge, -hedge, 0]
tse = [-hedge, -hedge, 0]
tsw = [-hedge, hedge, 0]
bnw = [hedge, hedge, -hedge]
bne = [hedge, -hedge, -hedge]
bse = [-hedge, -hedge, -hedge]
bsw = [-hedge, hedge, -hedge]

# north
triangles.append([tnw, tne, bne])
triangles.append([bne, bnw, tnw])
# triangles.append([tnw, tne, bne])

# west
triangles.append([tsw, tnw, bsw])
triangles.append([bnw, bsw, tnw])

# south
triangles.append([tsw, tse, bsw])
triangles.append([bse, bsw, tse])

# east
triangles.append([tne, tse, bse])
triangles.append([bse, bne, tne])


# bot
triangles.append([bnw, bne, bse])
triangles.append([bse, bsw, bnw])


triangles = np.asarray(triangles)

data = np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype)
data["vectors"] = triangles
name = "magnets"
the_mesh = mesh.Mesh(data.copy())
the_mesh.save(name + ".stl")
