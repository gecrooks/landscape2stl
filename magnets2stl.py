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



import ezdxf
from ezdxf.render.forms import cube, cylinder_2p
from ezdxf.addons.pycsg import CSG
from ezdxf.addons.meshex import stl_dumps

# create new DXF document
doc = ezdxf.new()
msp = doc.modelspace()

from landscape2stl import STLParameters

model = ezdxf.render.MeshBuilder()


# create same geometric primitives as MeshTransformer() objects
cube1 = cube()
cube1 = cube1.scale(20, 20, 10)


model_csg = CSG(cube1)


params = STLParameters()
magnet_radius = (params.magnet_diameter)/2 + params.magnet_padding
magnet_depth = params.magnet_depth + params.magnet_recess
magnet_sides = params.magnet_sides

pin_length = params.pin_length
pin_radius = (params.pin_diameter/2) + params.pin_padding
pin_sides = params.pin_sides


cylinder = cylinder_2p(
    count=magnet_sides,
    base_center=(0, -magnet_depth, 0),
    top_center=(0, magnet_depth, 0),
    radius=magnet_radius,
)
cylinder.translate(-5, 10, 0)
model_csg = model_csg - CSG(cylinder)

cylinder = cylinder_2p(
    count=magnet_sides,
    base_center=(0, -magnet_depth, 0),
    top_center=(0, magnet_depth, 0),
    radius=magnet_radius,
)
cylinder.translate(5, 10, 0)
model_csg = model_csg - CSG(cylinder)


cylinder = cylinder_2p(
    count=pin_sides,
    base_center=(0, -pin_length, 0),
    top_center=(0, pin_length, 0),
    radius=1,
)
cylinder.translate(-5,-10, 0)
model_csg = model_csg - CSG(cylinder)

cylinder = cylinder_2p(
    count=pin_sides,
    base_center=(0, -pin_length, 0),
    top_center=(0, pin_length, 0),
    radius=1,
)
cylinder.translate(5, -10, 0)
model_csg = model_csg - CSG(cylinder)


offset = params.bolt_hole_offset
sides = params.bolt_hole_sides
radius = params.bolt_hole_padding + params.bolt_hole_diameter/2 
depth = params.bolt_hole_depth
cylinder = cylinder_2p(
    count=sides, base_center=(0, 0, -depth), top_center=(0, 0, depth), radius=radius
)
cylinder.translate([0, 0, -5])
model_csg = model_csg - CSG(cylinder)



s = stl_dumps(model_csg.mesh())

filename = 'magnets.stl'

with open(filename, "w") as f:
        f.write(s)
