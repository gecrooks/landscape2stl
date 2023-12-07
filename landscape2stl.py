#!/usr/bin/env python

# Copyright 2023, Gavin E. Crooks and contributors
#
# This source code is licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
#
# landscape2stl: High resolution terrain models for 3D printing
#
# See README for more information
#
#
# Gavin E. Crooks 2023
#

# Note to self:
# latitude is north-south
# longitude is west-east (long way around)

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import ezdxf
import numpy as np
import py3dep
import xarray as xr
from ezdxf.addons.meshex import stl_dumpb
from ezdxf.addons.pycsg import CSG
from ezdxf.render.forms import cylinder_2p
from numpy.typing import ArrayLike
from typing_extensions import TypeAlias

# Many units and coordinate systems. Use TypeAlias's in desperate effort to
# keep everything straight
MM: TypeAlias = float  # millimeters
Meters: TypeAlias = float  # meters
Degrees: TypeAlias = float
ECEF: TypeAlias = tuple[
    Meters, Meters, Meters
]  # Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates
LLA: TypeAlias = tuple[
    Degrees, Degrees, Meters
]  # latitude, longitude, altitude (in meters) coordinates
ENU: TypeAlias = tuple[MM, MM, MM]  # East, North, Up model coordinates (millimeters)
BBox: TypeAlias = tuple[
    Degrees, Degrees, Degrees, Degrees
]  # Geographic bounding box: south, west, north, east


# Various presets featuring varied terrains for testing and development.
presets: dict[str, tuple[BBox, int]] = {
    "half_dome": ((37.72, -119.57, 37.76, -119.52), 24_000), # 37.75°N 119.53°W
    "west_of_half_dome": ((37.72, -119.62, 37.76, -119.57), 24_000),
    "whitney": ((36.56, -118.33, 36.60, -118.28), 24_000),  # High point test
    "yosemite_west": ((37.70, -119.80, 37.80, -119.65), 62_500),
    "yosemite_valley": ((37.70, -119.65, 37.80, -119.50), 62_500),
    "clouds_rest": ((37.70, -119.50, 37.80, -119.35), 62_500),
    "grand_canyon": ((36.0000, -112.2000, 36.2000, -111.9500), 125_000),
    "shasta": ((41.3000, -122.3500, 41.5000, -122.0500), 125_000),
    "shasta_west": ((41.3000, -122.6500, 41.5000, -122.3500), 125_000),
    "joshua_tree": ((33.8, -116.0000, 34.0000, -115.75), 125_000),
    "owens_valley": ((36.5, -120, 38, -118), 1_000_000),
    "denali": ((62.5000, -152.0000, 63.5000, -150.0000), 1_000_000),
    # "cali": ((32, -125., 42, -114),  1_000_000),
}


standard_scales = [
    24_000,  # 1" = 2000', about 2.5" to 1 mile
    62_500,  # about 1" to 1 mile
    125_000,  # about 1" to 2 miles
    250_000,  # about 1" to 4 miles
    500_000,  # about 1" to 8 miles
    1_000_000,  # about 1" to 16 miles
]


default_cache = "cache"


@dataclass
class STLParameters:
    scale: int = 62_500
    resolution: int = 10  # meters
    resolution_choices: tuple[int] = (10, 30)
    pitch: MM = 0.20  # Half the minimum feature size

    min_altitude: Meters = -100.0  # Lowest point in US is -86 m

    drop_sea_level: bool = True
    sea_level: Meters = 1.0  # 0.01
    sea_level_drop: MM = 0.4

    exaggeration: float = 1.0

    base_height: MM = 10.0

    magnet_holes: bool = True
    magnet_spacing: Degrees = 0.0
    magnet_diameter: MM = 6.00
    magnet_padding: MM = 0.025
    magnet_depth: MM = 2.00
    magnet_recess: MM = 0.05
    magnet_sides: int = 24

    pin_holes: bool = True
    pin_length: MM = 9
    pin_diameter: MM = 1.75
    pin_padding: MM = 0.05
    pin_sides: int = 8

    bolt_holes: bool = True
    bolt_hole_offset: MM = 10
    bolt_hole_diameter: MM = 5.6
    bolt_hole_padding: MM = 0.2
    bolt_hole_depth: MM = 9.1
    bolt_hole_sides: int = 24

    def __post_init__(self):
        default_magnet_spacing = {
            24_000: 0.01,
            62_500: 0.025,
            125_000: 0.05,
            250_000: 0.10,
            500_000: 0.25,  # ???
            1_000_000: 0.50,  # ???
        }

        if not self.magnet_spacing:
            if self.scale in default_magnet_spacing:
                self.magnet_spacing = default_magnet_spacing[self.scale]

# end STLParameters


def main() -> int:
    default_params = STLParameters()
    parser = argparse.ArgumentParser(description="Create landscape STLs")
    parser.add_argument(
        "coordinates",
        metavar="S W N E",
        type=float,
        nargs="*",
        help="Latitude/longitude coordinates for slab (Order south edge, west edge, north edge, east edge)",
    )

    parser.add_argument("--preset", dest="preset", choices=presets.keys())

    parser.add_argument("--scale", dest="scale", type=int, help="Map scale")

    parser.add_argument(
        "--exaggeration",
        dest="exaggeration",
        type=float,
        default=1.0,
        help="Vertical exaggeration",
    )

    parser.add_argument(
        "--resolution", dest="resolution", default=default_params.resolution, choices=default_params.resolution_choices, type=int, help="DEM resolution"
    )

    parser.add_argument(
        "--magnets", dest="magnets", type=float, help="Magnet spacing (in degrees)"
    )

    parser.add_argument("--name", dest="name", type=str, help="Filename for model")

    parser.add_argument("-v", "--verbose", action="store_true")

    # TODO: Add cache argument

    args = vars(parser.parse_args())
    name = None

    if args["preset"] is not None:
        if args["coordinates"]:
            parser.print_help()
            return 1

        name = args["preset"]
        args["coordinates"] = presets[name][0]
        if args["scale"] is None:
            args["scale"] = presets[name][1]

        if args["name"] is None:
            args["name"] = args["preset"]

    if not args["coordinates"]:
        parser.print_help()
        return 0

    if args["scale"] is None:
        args["scale"] = default_params.scale

    params = STLParameters(
        scale=args["scale"],
        exaggeration=args["exaggeration"],
        magnet_spacing=args["magnets"],
        resolution=args["resolution"],
    )

    create_stl(params, args["coordinates"], name=args["name"], verbose=args["verbose"])

    return 0


def create_stl(
    params: STLParameters,
    boundary: BBox,
    name: Optional[str] = None,
    verbose: bool = False,
) -> None:
    # Locate origin
    south, west, north, east = boundary
    origin = locate_origin(boundary)

    # Calculate steps...
    north_west_enu = lla_to_model((north, west, 0.0), origin, params)
    south_east_enu = lla_to_model((south, east, 0.0), origin, params)

    extent_ns =  south_east_enu[0] - north_west_enu[0]
    extent_we = north_west_enu[1] - south_east_enu[1]

    ns_steps = int(round(extent_ns / params.pitch))
    we_steps = int(round(extent_we / params.pitch))
    steps = max(ns_steps, we_steps)

    elevation = download_elevation(boundary, steps, params.resolution, verbose)

    if verbose:
        print("Building terrain...")

    surface = elevation_to_surface(elevation, origin, params)

    # Add a little bit of noise. Hack for smooth seascapes
    # FIXME: Do I still need this hack?
    surface += 0.001 * np.random.uniform(size=surface.shape)

    if verbose:
        print("Triangulating surface...")

    surface_mesh = triangulate_surface(surface, boundary, origin, params)

    if verbose:
        print("Triangulating base...")

    base_mesh = triangulate_base(boundary, origin, params, steps)

    model = surface_mesh
    model.add_mesh(mesh=base_mesh)

    # model = base_mesh

    model = model.optimize_vertices()
    model.normalize_faces()

    if verbose:
        print("Faces:", len(model.faces))

    if verbose:
        print("Creating STL...")

    if name is None:
        filename = "{:.2f}_{:.2f}_{:.2f}_{:.2f}.stl".format(*boundary)
    else:
        filename = name + ".stl"

    binary_stl = stl_dumpb(model)

    if verbose:
        print(f"Saving {filename}")

    with open(filename, "wb") as binary_file:
        binary_file.write(binary_stl)


# end create_stl


def locate_origin(boundary: BBox) -> LLA:
    # Locate a point in the middle of the landscape tile which defines the
    # local up direction. Model distances are defined relative to this point

    south, west, north, east = boundary

    origin = (south + north) / 2, (east + west) / 2, -100000.0

    # adjust origin so that 4 corners are equal height

    alt = -100
    west_north = lla_to_ecef((north, west, alt))
    east_north = lla_to_ecef((north, east, alt))
    east_south = lla_to_ecef((south, east, alt))
    west_south = lla_to_ecef((south, west, alt))
    new_origin_ecef = (
        (west_north[0] + east_south[0] + west_south[0] + east_north[0]) / 4,
        (west_north[1] + east_south[1] + west_south[1] + east_north[1]) / 4,
        (west_north[2] + east_south[2] + west_south[2] + east_north[2]) / 4,
    )
    new_origin = ecef_to_lla(new_origin_ecef)

    # Honestly, dunno why this works.
    origin = (origin[0] * 2 - new_origin[0], new_origin[1], 0.0)

    return origin


def download_elevation(
    boundary: BBox,
    steps: int,
    resolution: int,
    verbose: bool = False,
) -> xr.Dataset | xr.DataArray:
    elevation: xr.Dataset | xr.DataArray
    south, west, north, east = boundary

    xcoords = np.linspace(west, east, steps)
    ycoords = np.linspace(south, north, steps)

    filename = "{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.nc".format(steps, *boundary)

    if not os.path.exists(default_cache):
        os.mkdir(default_cache)
    fname = os.path.join(default_cache, filename)

    if not os.path.exists(fname):
        if verbose:
            print("Downloading elevation data... ", end="", flush=True)
        elevation = py3dep.elevation_bygrid(
            tuple(xcoords), tuple(ycoords), crs="EPSG:4326", resolution=resolution
        )  # or 30

        elevation.to_netcdf(fname)
        if verbose:
            print("Done", flush=True)

    if verbose:
        print("Loading elevation data from cache...", end="", flush=True)
    elevation = xr.open_dataset(fname)
    if verbose:
        print("", flush=True)

    return elevation


def elevation_to_surface(
    elevation: xr.Dataset | xr.DataArray, origin: LLA, params: STLParameters
) -> np.ndarray:
    ycoords = np.asarray(elevation.coords["y"])
    xcoords = np.asarray(elevation.coords["x"])
    steps = len(ycoords)
    elevation_array = np.asarray(elevation.to_array()).reshape((steps, steps)).T

    # Missing date will be nan
    elevation_array = np.nan_to_num(elevation_array, nan=0.0)

    if params.drop_sea_level:
        dropped_sea_level = (
            -(params.scale * params.sea_level_drop / 1000) / params.exaggeration
        )
        elevation_array[elevation_array <= params.sea_level] = dropped_sea_level

    surface = np.zeros(shape=(steps, steps, 3))

    for x in range(steps):
        for y in range(steps):
            lat = ycoords[y]
            lon = xcoords[x]
            alt = elevation_array[x, y]
            surface[x, y] = lla_to_model((lat, lon, alt), origin, params)

    return surface


def triangulate_surface(
    surface: np.ndarray,
    boundary: BBox,
    origin: LLA,
    params: STLParameters,
) -> ezdxf.render.MeshBuilder:
    model = ezdxf.render.MeshBuilder()
    steps = surface.shape[0]
    south, west, north, east = boundary

    # Top surface
    for x in range(steps - 1):
        for y in range(steps - 1):
            if ((x + y) % 2) == 0:
                model.add_face(
                    [surface[x, y], surface[x + 1, y], surface[x + 1, y + 1]]
                )
                model.add_face(
                    [surface[x, y], surface[x + 1, y + 1], surface[x, y + 1]]
                )
            else:
                model.add_face([surface[x, y + 1], surface[x, y], surface[x + 1, y]])
                model.add_face(
                    [surface[x + 1, y], surface[x + 1, y + 1], surface[x, y + 1]]
                )

    bot_corners = corners_to_model(boundary, params.min_altitude, origin, params)
    west_north_bot, west_south_bot, east_south_bot, east_north_bot = bot_corners

    bot_height = west_north_bot[2]

    # south
    xcoords = np.linspace(west_south_bot[0], east_south_bot[0], steps)
    ycoords = np.linspace(west_south_bot[1], east_south_bot[1], steps)
    for x in range(steps - 1):
        y = 0
        # Sloped edge
        tri = (
            surface[x, y],
            (xcoords[x], ycoords[x], bot_height),
            (xcoords[x + 1], ycoords[x + 1], bot_height),
        )

        model.add_face(tri)
        tri = (
            surface[x, y],
            (xcoords[x + 1], ycoords[x + 1], bot_height),
            surface[x + 1, y],
        )
        model.add_face(tri)

    # north
    xcoords = np.linspace(west_north_bot[0], east_north_bot[0], steps)
    ycoords = np.linspace(west_north_bot[1], east_north_bot[1], steps)

    for x in range(steps - 1):
        y = -1
        tri = (
            surface[x, y],
            (xcoords[x], ycoords[x], bot_height),
            (xcoords[x + 1], ycoords[x + 1], bot_height),
        )
        model.add_face(tri)
        tri = (
            surface[x, y],
            (xcoords[x + 1], ycoords[x + 1], bot_height),
            surface[x + 1, y],
        )
        model.add_face(tri)

    # west
    xcoords = np.linspace(west_south_bot[0], west_north_bot[0], steps)
    ycoords = np.linspace(west_south_bot[1], west_north_bot[1], steps)
    for s in range(steps - 1):
        x = 0
        tri = (
            surface[x, s],
            (xcoords[s], ycoords[s], bot_height),
            (xcoords[s + 1], ycoords[s + 1], bot_height),
        )
        model.add_face(tri)
        tri = (
            surface[x, s],
            (xcoords[s + 1], ycoords[s + 1], bot_height),
            surface[x, s + 1],
        )
        model.add_face(tri)

    # east
    xcoords = np.linspace(east_south_bot[0], east_north_bot[0], steps)
    ycoords = np.linspace(east_south_bot[1], east_north_bot[1], steps)
    for s in range(steps - 1):
        x = -1
        tri = (
            surface[x, s],
            (xcoords[s], ycoords[s], bot_height),
            (xcoords[s + 1], ycoords[s + 1], bot_height),
        )
        model.add_face(tri)
        tri = (
            surface[x, s],
            (xcoords[s + 1], ycoords[s + 1], bot_height),
            surface[x, s + 1],
        )
        model.add_face(tri)

    return model


def triangulate_base(
    boundary: BBox,
    origin: LLA,
    params: STLParameters,
    steps: int,
) -> ezdxf.render.MeshBuilder:
    model = ezdxf.render.MeshBuilder()
    south, west, north, east = boundary
    steps = steps//8

    base_alt = params.min_altitude - (
        params.base_height * params.scale / (1000 * params.exaggeration)
    )

    magnets_alt = (base_alt + params.min_altitude) / 2

    top_corners = corners_to_model(boundary, params.min_altitude, origin, params)
    west_north_top, west_south_top, east_south_top, east_north_top = top_corners

    bot_corners = corners_to_model(boundary, base_alt, origin, params)
    west_north_bot, west_south_bot, east_south_bot, east_north_bot = bot_corners

    mag_corners = corners_to_model(boundary, magnets_alt, origin, params)
    west_north_mag, west_south_mag, east_south_mag, east_north_mag = mag_corners

    bot = (
        west_south_bot[2] + east_south_bot[2] + east_north_bot[2] + west_north_bot[2]
    ) / 4

    westing = np.linspace(west, east, steps)
    northing = np.linspace(north, south, steps)

    # South
    south_top = [
        lla_to_model((south, w, params.min_altitude), origin, params) for w in westing
    ]
    south_base = [lla_to_model((south, w, base_alt), origin, params) for w in westing]
    south_bot = [
        find_point_on_line(llat, llab, bot) for llat, llab in zip(south_top, south_base)
    ]

    for i in range(steps - 1):
        model.add_face([south_top[i], south_bot[i], south_top[i + 1]])
        model.add_face([south_top[i + 1], south_bot[i], south_bot[i + 1]])

    # North
    north_top = [
        lla_to_model((north, w, params.min_altitude), origin, params) for w in westing
    ]
    north_base = [lla_to_model((north, w, base_alt), origin, params) for w in westing]
    north_bot = [
        find_point_on_line(llat, llab, bot) for llat, llab in zip(north_top, north_base)
    ]

    for i in range(steps - 1):
        model.add_face([north_top[i], north_bot[i], north_top[i + 1]][::-1])
        model.add_face([north_top[i + 1], north_bot[i], north_bot[i + 1]][::-1])

    # East
    east_top = [
        lla_to_model((n, east, params.min_altitude), origin, params) for n in northing
    ]
    east_base = [lla_to_model((n, east, base_alt), origin, params) for n in northing]
    east_bot = [
        find_point_on_line(llat, llab, bot) for llat, llab in zip(east_top, east_base)
    ]

    for i in range(steps - 1):
        model.add_face([east_top[i], east_bot[i], east_top[i + 1]][::-1])
        model.add_face([east_top[i + 1], east_bot[i], east_bot[i + 1]][::-1])

    # West
    west_top = [
        lla_to_model((n, west, params.min_altitude), origin, params) for n in northing
    ]
    west_base = [lla_to_model((n, west, base_alt), origin, params) for n in northing]
    west_bot = [
        find_point_on_line(llat, llab, bot) for llat, llab in zip(west_top, west_base)
    ]

    for i in range(steps - 1):
        model.add_face([west_top[i], west_bot[i], west_top[i + 1]])
        model.add_face([west_top[i + 1], west_bot[i], west_bot[i + 1]])


    # bot of base
    for i in range(steps - 1):
        model.add_face([north_bot[i], south_bot[i], north_bot[i + 1]][::-1])
        model.add_face([north_bot[i + 1], south_bot[i], south_bot[i + 1]][::-1])

    def make_hole(sides, depth, radius, center, axis):
        w = axis
        v = (0, 0, 1)

        v_normalized = normalize(v)
        w_normalized = normalize(w)

        rot_axis = np.cross(v_normalized, w_normalized)
        rot_axis = normalize(rot_axis)

        theta = angle_between(v_normalized, w_normalized)

        hole = cylinder_2p(
            count=sides,
            base_center=(0, 0, -depth),
            top_center=(0, 0, depth),
            radius=radius,
        )
        hole.rotate_axis(rot_axis, theta)
        hole.translate(*center)

        return hole

    model_csg = CSG(model)

    magnets = params.magnet_spacing
    long_steps = 1 + round((east - west) / magnets) * 2
    lat_steps = 1 + round((north - south) / magnets) * 2
    longs = np.linspace(west, east, long_steps)
    lats = np.linspace(south, north, lat_steps)

    south_normal = triangle_normal(east_south_bot, east_south_top, west_south_bot)
    north_normal = triangle_normal(east_north_bot, west_north_bot, east_north_top)
    west_normal = triangle_normal(west_south_bot, west_south_top, west_north_top)
    east_normal = triangle_normal(east_south_top, east_south_bot, east_north_top)

    if params.magnet_holes:

        mag_radius = (params.magnet_diameter) / 2 + params.magnet_padding
        mag_depth = params.magnet_depth + params.magnet_recess
        mag_sides = params.magnet_sides

        # south
        for i in range(1, long_steps, 2):
            mag_lla = (south, longs[i], magnets_alt)
            mag_enu = lla_to_model(mag_lla, origin, params)
            hole = make_hole(mag_sides, mag_depth, mag_radius, mag_enu, south_normal)
            model_csg = model_csg - CSG(hole)

        # north
        for i in range(1, long_steps, 2):
            mag_lla = (north, longs[i], magnets_alt)
            mag_enu = lla_to_model(mag_lla, origin, params)
            hole = make_hole(mag_sides, mag_depth, mag_radius, mag_enu, north_normal)
            model_csg = model_csg - CSG(hole)

        # west
        for i in range(1, lat_steps, 2):
            mag_lla = (lats[i], west, magnets_alt)
            mag_enu = lla_to_model(mag_lla, origin, params)
            hole = make_hole(mag_sides, mag_depth, mag_radius, mag_enu, west_normal)
            model_csg = model_csg - CSG(hole)

        # east
        for i in range(1, lat_steps, 2):
            mag_lla = (lats[i], east, magnets_alt)
            mag_enu = lla_to_model(mag_lla, origin, params)
            hole = make_hole(mag_sides, mag_depth, mag_radius, mag_enu, east_normal)
            model_csg = model_csg - CSG(hole)

    if params.pin_holes:

        pin_length = params.pin_length
        pin_radius = (params.pin_diameter / 2) + params.pin_padding
        pin_sides = params.pin_sides

        # south
        for i in range(2, long_steps - 1, 2):
            pin_lla = (south, longs[i], magnets_alt)
            pin_enu = lla_to_model(pin_lla, origin, params)
            hole = make_hole(pin_sides, pin_length, pin_radius, pin_enu, south_normal)
            model_csg = model_csg - CSG(hole)

        # north
        for i in range(2, long_steps - 1, 2):
            pin_lla = (north, longs[i], magnets_alt)
            pin_enu = lla_to_model(pin_lla, origin, params)
            hole = make_hole(pin_sides, pin_length, pin_radius, pin_enu, north_normal)
            model_csg = model_csg - CSG(hole)

        # west
        for i in range(2, lat_steps - 1, 2):
            pin_lla = (lats[i], west, magnets_alt)
            pin_enu = lla_to_model(pin_lla, origin, params)
            hole = make_hole(pin_sides, pin_length, pin_radius, pin_enu, west_normal)
            model_csg = model_csg - CSG(hole)

        # east
        for i in range(2, lat_steps - 1, 2):
            pin_lla = (lats[i], east, magnets_alt)
            pin_enu = lla_to_model(pin_lla, origin, params)
            hole = make_hole(pin_sides, pin_length, pin_radius, pin_enu, east_normal)
            model_csg = model_csg - CSG(hole)

    if params.bolt_holes:
        # Corner bolt holes
        offset = params.bolt_hole_offset
        sides = params.bolt_hole_sides
        radius = params.bolt_hole_padding + params.bolt_hole_diameter / 2
        depth = params.bolt_hole_depth
        cylinder = cylinder_2p(
            count=sides,
            base_center=(0, 0, -depth),
            top_center=(0, 0, depth),
            radius=radius,
        )
        cylinder.translate(west_north_bot)
        cylinder.translate([offset, -offset, 0])
        model_csg = model_csg - CSG(cylinder)

        cylinder = cylinder_2p(
            count=sides,
            base_center=(0, 0, -depth),
            top_center=(0, 0, depth),
            radius=radius,
        )
        cylinder.translate(west_south_bot)
        cylinder.translate([offset, offset, 0])
        model_csg = model_csg - CSG(cylinder)

        cylinder = cylinder_2p(
            count=sides,
            base_center=(0, 0, -depth),
            top_center=(0, 0, depth),
            radius=radius,
        )
        cylinder.translate(east_south_bot)
        cylinder.translate([-offset, offset, 0])
        model_csg = model_csg - CSG(cylinder)

        cylinder = cylinder_2p(
            count=sides,
            base_center=(0, 0, -depth),
            top_center=(0, 0, depth),
            radius=radius,
        )
        cylinder.translate(east_north_bot)
        cylinder.translate([-offset, -offset, 0])
        model_csg = model_csg - CSG(cylinder)

        center = (np.asarray(east_north_bot) + np.asarray(west_south_bot)) / 2.0
        cylinder = cylinder_2p(
            count=sides,
            base_center=(0, 0, -depth),
            top_center=(0, 0, depth),
            radius=radius,
        )
        cylinder.translate(center)
        model_csg = model_csg - CSG(cylinder)

    model = model_csg.mesh()
    return model


# End triangulate base


def triangle_normal(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> np.ndarray:
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)

    AB = B - A
    AC = C - A

    normal = np.cross(AB, AC)
    normal_normalized = normal / np.linalg.norm(normal)

    return normal_normalized


def lla_to_ecef(lat_lon_alt: LLA) -> ECEF:
    """
    Convert latitude, longitude, altitude (LLA) coordinates
    to Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates.
    """
    latitude, longitude, altitude = lat_lon_alt

    # Constants for WGS84
    a = 6378137.0  # Equatorial radius (meters)
    e = 0.08181919084  # Eccentricity

    # Convert latitude and longitude to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Calculate prime vertical radius of curvature
    N = a / math.sqrt(1 - e**2 * math.sin(lat_rad) ** 2)

    # Calculate ECEF coordinates
    X = (N + altitude) * math.cos(lat_rad) * math.cos(lon_rad)
    Y = (N + altitude) * math.cos(lat_rad) * math.sin(lon_rad)
    Z = ((1 - e**2) * N + altitude) * math.sin(lat_rad)

    return X, Y, Z


def ecef_to_lla(ecef: ECEF) -> LLA:
    """
    Convert to Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates
    to latitude, longitude, altitude (LLA) coordinates.
    """

    x, y, z = ecef

    # Constants for WGS84
    a = 6378137.0  # Semi-major axis
    e_sq = 0.00669437999014  # Square of eccentricity

    # Calculate longitude
    lon = math.atan2(y, x)

    # Iterative calculation of latitude and altitude
    p = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, p * (1.0 - e_sq))

    while True:
        N = a / math.sqrt(1 - e_sq * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - N
        new_lat = math.atan2(z, p * (1.0 - e_sq * N / (N + alt)))

        if abs(new_lat - lat) < 1e-9:
            lat = new_lat
            break

        lat = new_lat

    # Convert from radians to degrees
    lon_deg = math.degrees(lon)
    lat_deg = math.degrees(lat)

    return lat_deg, lon_deg, alt


def lla_to_enu(lat_lon_alt: LLA, origin_lat_lon_alt: LLA) -> ENU:
    """
    Convert latitude, longitude, altitude (LLA) coordinates
    to local Cartesian coordinates (East-North-Up or ENU) relative
    to a given origin point. In millimeters
    """
    # Convert origin to ECEF coordinates
    x_target, y_target, z_target = lla_to_ecef(lat_lon_alt)
    x_origin, y_origin, z_origin = lla_to_ecef(origin_lat_lon_alt)

    # Calculate ECEF vector between origin and target point
    dx, dy, dz = x_target - x_origin, y_target - y_origin, z_target - z_origin

    # Convert origin latitude and longitude to radians
    lat_rad = math.radians(origin_lat_lon_alt[0])
    lon_rad = math.radians(origin_lat_lon_alt[1])

    # Define the rotation matrix
    R = np.array(
        [
            [-math.sin(lon_rad), math.cos(lon_rad), 0],
            [
                -math.sin(lat_rad) * math.cos(lon_rad),
                -math.sin(lat_rad) * math.sin(lon_rad),
                math.cos(lat_rad),
            ],
            [
                math.cos(lat_rad) * math.cos(lon_rad),
                math.cos(lat_rad) * math.sin(lon_rad),
                math.sin(lat_rad),
            ],
        ]
    )

    # Multiply the rotation matrix by the ECEF vector
    enu = R.dot(np.array([dx, dy, dz]))
    enu *= 1000  # meters to mm

    return enu[0], enu[1], enu[2]


def lla_to_model(
    lat_lon_alt: LLA, origin_lat_lon_alt: LLA, params: STLParameters
) -> ENU:
    """
    Convert latitude, longitude, and altitude (LLA) coordinates
    to model ENU Cartesian coordinates in millimeters
    """

    lat, lon, alt = lat_lon_alt
    enu = lla_to_enu((lat, lon, alt * params.exaggeration), origin_lat_lon_alt)
    enu_scaled = np.asarray(enu)
    enu_scaled /= params.scale

    return (enu_scaled[0], enu_scaled[1], enu_scaled[2])


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def angle_between(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))


def corners_to_model(
    boundary: BBox,
    alt: Meters,
    origin: LLA,
    params: STLParameters,
) -> tuple[ENU, ENU, ENU, ENU]:
    south, west, north, east = boundary
    west_north: ENU = lla_to_model((north, west, alt), origin, params)
    east_north: ENU = lla_to_model((north, east, alt), origin, params)
    east_south: ENU = lla_to_model((south, east, alt), origin, params)
    west_south: ENU = lla_to_model((south, west, alt), origin, params)

    return west_north, west_south, east_south, east_north


def find_point_on_line(p1, p2, z3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    if z2 == z1:  # Avoid division by zero
        raise ValueError("Points p1 and p2 have the same z coordinate.")

    x3 = x1 + (x2 - x1) * ((z3 - z1) / (z2 - z1))
    y3 = y1 + (y2 - y1) * ((z3 - z1) / (z2 - z1))

    return x3, y3, z3


if __name__ == "__main__":
    sys.exit(main())
