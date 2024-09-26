#!/usr/bin/env python

# Copyright 2023-2024, Gavin E. Crooks
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

# Disclaimer: This is ugly hacked together prototype code. You have been warned.

# Note to self:
# latitude is north-south
# longitude is west-east (long way around)

import argparse
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple

import ezdxf
import numpy as np
import pandas as pd
import py3dep
import requests
import us
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
    # "half_dome": ((37.72, -119.57, 37.76, -119.52), 24_000), # 37.75°N 119.53°W
    "half_dome": ((37.72, -119.57, 37.77, -119.51), 24_000),  # 37.75°N 119.53°W
    "west_of_half_dome": ((37.72, -119.62, 37.76, -119.57), 24_000),
    "whitney": ((36.56, -118.33, 36.60, -118.28), 24_000),  # High point test
    "grand_canyon": ((36.0000, -112.2000, 36.2000, -111.9500), 125_000),
    "shasta": ((41.3000, -122.3500, 41.5000, -122.0500), 125_000),
    "shasta_west": ((41.3000, -122.6500, 41.5000, -122.3500), 125_000),
    "joshua_tree": ((33.8, -116.0000, 34.0000, -115.75), 125_000),
    "owens_valley": ((36.5, -120, 38, -118), 1_000_000),
    "denali": ((62.5000, -152.0000, 63.5000, -150.0000), 1_000_000),
}


# standard_scales = [
#     24_000,  # 1" = 2000', about 2.5" to 1 mile
#     62_500,  # about 1" to 1 mile
#     125_000,  # about 1" to 2 miles
#     250_000,  # about 1" to 4 miles
#     500_000,  # about 1" to 8 miles
#     1_000_000,  # about 1" to 16 miles
# ]


default_cache = "cache"


@dataclass
class STLParameters:
    """Parameters for the STL terrain models (apart from actual coordinates).
    Scale is the important parameter to set, the rest can generally be left at default.
    """

    scale: int = 62_500
    resolution: int = 0  # Auto set in __post_init__
    resolution_choices: tuple[int] = (10, 30)  # meters
    pitch: MM = 0.40  # Nozzle size

    min_altitude: Meters = -100.0  # Lowest point in US is -86 m

    drop_sea_level: bool = True
    sea_level: Meters = 1.7
    sea_level_drop: MM = 0.48  # 6 layers
    exaggeration: float = 0.0  # Auto set in __post_init__

    base_height: MM = 10.0

    magnet_holes: bool = True
    magnet_spacing: Degrees = 0.0  # Auto set in __post_init__
    magnet_diameter: MM = 6.00
    magnet_padding: MM = 0.025
    magnet_depth: MM = 2.00
    magnet_recess: MM = 0.10
    magnet_sides: int = 24

    pin_holes: bool = True
    pin_length: MM = 9
    pin_diameter: MM = 1.75
    pin_padding: MM = 0.05 * 3
    pin_sides: int = 8

    bottom_holes: bool = False
    bottom_hole_offset: MM = 10
    bottom_hole_diameter: MM = 5.6
    bottom_hole_padding: MM = 0.2
    bottom_hole_depth: MM = 9.1
    bottom_hole_sides: int = 24

    def __post_init__(self):
        if not self.magnet_spacing:
            if self.scale == 24_000:
                self.magnet_spacing = 1 / 64
            else:
                self.magnet_spacing = self.scale / 2_000_000

        if not self.resolution:
            if self.scale < 250_000:
                self.resolution = self.resolution_choices[0]
            else:
                self.resolution = self.resolution_choices[1]

        if not self.exaggeration:
            # Heuristic for vertical exaggeration
            # scale exaggeration
            # <= 62_500     1.0
            # 125_000       1.5
            # 250_000       2.0
            # 500_000       2.5
            # 1_000_000     3.0
            if self.scale <= 62_500:
                self.exaggeration = 1.0
            else:
                self.exaggeration = 3 - 0.5 * math.log2(1_000_000 / self.scale)


# end STLParameters


def main() -> int:
    default_params = STLParameters()
    parser = argparse.ArgumentParser(description="Create quadrangle landscape STLs")
    parser.add_argument(
        "coordinates",
        metavar="S W N E",
        type=float,
        nargs="*",
        help="Latitude/longitude coordinates for quadrangle (Order south edge, west edge, north edge, east edge)",
    )

    parser.add_argument("--preset", dest="preset", choices=presets.keys())

    parser.add_argument("--quad", dest="quad", type=str)

    parser.add_argument("--state", dest="state", type=str, default="CA")

    parser.add_argument("--scale", dest="scale", type=int, help="Map scale")

    parser.add_argument(
        "--exaggeration",
        dest="exaggeration",
        type=float,
        # default=1.0,
        help="Vertical exaggeration",
    )

    # parser.add_argument(
    #     "--resolution",
    #     dest="resolution",
    #     default=default_params.resolution,
    #     choices=default_params.resolution_choices,
    #     type=int,
    #     help="DEM resolution",
    # )

    parser.add_argument(
        "--magnets", dest="magnets", type=float, help="Magnet spacing (in degrees)"
    )

    parser.add_argument("--name", dest="name", type=str, help="Filename for model")

    parser.add_argument("-v", "--verbose", action="store_true")

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

    if args["quad"] is not None:
        name = args["quad"].lower().replace(" ", "_")
        coords = quad_coordinates(name, args["state"])
        args["coordinates"] = coords
        # args["scale"] = 62_500 # params default to this scale

        if args["name"] is None:
            args["name"] = "quad_" + args["state"].lower() + "_" + name

    if not args["coordinates"]:
        parser.print_help()
        return 0

    if args["scale"] is None:
        args["scale"] = default_params.scale

    params = STLParameters(
        scale=args["scale"],
        exaggeration=args["exaggeration"],
        magnet_spacing=args["magnets"],
        # resolution=args["resolution"],
        # projection=args["projection"],
    )

    create_stl(
        params, args["coordinates"], filename=args["name"], verbose=args["verbose"]
    )

    return 0


def ustopo_current():
    url = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/Metadata/ustopo_current.zip"
    zip_file_name = "ustopo_current.zip"
    csv_file_name = "ustopo_current.csv"
    directory = "cache"

    zip_file_path = os.path.join(directory, zip_file_name)

    if not os.path.exists(zip_file_path):
        os.makedirs(directory, exist_ok=True)
        response = requests.get(url)
        with open(zip_file_path, "wb") as f:
            f.write(response.content)

    with zipfile.ZipFile(zip_file_path, "r") as z:
        with z.open(csv_file_name) as csv_file:
            df = pd.read_csv(csv_file)

    return df


def quad_coordinates(quad_name, state="CA"):
    df = ustopo_current()

    quad_name = quad_name.lower().replace("_", " ")

    condition = (df["map_name"].str.lower() == quad_name) & df[
        "state_list"
    ].str.contains(state)

    row = df[condition]

    if len(row) == 0:
        raise ValueError("Quadrangle " + quad_name + " not found")

    southbc = row["southbc"].astype(float).iloc[0]
    westbc = row["westbc"].astype(float).iloc[0]

    return southbc, westbc, southbc + 1 / 8, westbc + 1 / 8


def quad_from_coordinates(lat, long):
    df = ustopo_current()

    condition = (
        (df["southbc"].astype(float) <= lat)
        & (lat < df["northbc"].astype(float))
        & (df["westbc"].astype(float) <= long)
        & (long < df["eastbc"].astype(float))
    )

    row = df[condition]
    if len(row) == 0:
        return (None, None)
    name = row["map_name"].astype(str).iloc[0]
    state_name = row["primary_state"].astype(str).iloc[0]
    state = us.states.lookup(state_name)

    return name, state.abbr


def create_quad_stl(name, state, filename=None, verbose=False):
    coords = quad_coordinates(name, state)
    if filename is None:
        filename = "quad_" + state.lower() + "_" + name.lower().replace(" ", "_")

    params = STLParameters(
        scale=62_500,
    )

    create_stl(params, coords, filename, verbose)


def create_stl(
    params: STLParameters,
    boundary: BBox,
    filename: Optional[str] = None,
    verbose: bool = False,
) -> None:
    if verbose:
        print(params)

    # Locate origin
    south, west, north, east = boundary
    origin = (south + north) / 2, (east + west) / 2, 0.0

    # Calculate steps...
    north_west_enu = lla_to_model((north, west, 0.0), origin, params)
    south_east_enu = lla_to_model((south, east, 0.0), origin, params)

    extent_ns = south_east_enu[0] - north_west_enu[0]
    extent_we = north_west_enu[1] - south_east_enu[1]

    ns_steps = int(round(extent_ns / params.pitch))
    we_steps = int(round(extent_we / params.pitch))
    steps = max(ns_steps, we_steps)

    elevation = download_elevation(boundary, steps, params.resolution, verbose)

    if verbose:
        print("Building terrain...")

    surface = elevation_to_surface(elevation, origin, params)

    # Add a little bit of noise. Hack for smooth seascapes
    # (I think this is necessary to make sure we don't have large number of co-planer triangles
    # Which seems to upset the CSG module)
    # FIXME: Do I still need this hack?
    # surface += 0.001 * np.random.uniform(size=surface.shape)

    if verbose:
        print("Triangulating surface...")

    # Another hack: We build the surface and the base separately. The base
    # alone uses Constructive Solid Geometry (CSG). The CSG module in ezdxf is using
    # a binary space partitioning (BSP) tree. This shatters triangles into lots of
    # sub-triangles. So if we tried using CSG on our landscape surface that has 100s of
    # thousands of initial triangles we end up with millions.

    # Also even with small models ezdxf generates non-manifold meshes that can upset the slicer.
    # Errors seem to be reparable or ignorable (mostly).

    surface_mesh = triangulate_surface(surface, boundary, origin, params)

    if verbose:
        print("Triangulating base...")

    base_mesh = triangulate_base(boundary, origin, params, steps)

    model = surface_mesh
    model.add_mesh(mesh=base_mesh)

    model = model.optimize_vertices()
    model.normalize_faces()

    if verbose:
        print("Faces:", len(model.faces))

    if verbose:
        print("Creating STL...")

    if filename is None:
        filename = "{:f}_{:f}_{:f}_{:f}.stl".format(*boundary)
    else:
        filename = filename + ".stl"

    binary_stl = stl_dumpb(model)

    if verbose:
        print(f"Saving {filename}")

    with open(filename, "wb") as binary_file:
        binary_file.write(binary_stl)


# end create_stl


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

    # print("Elevation min max", np.min(elevation_array), np.max(elevation_array))

    # Missing date will be nan
    elevation_array = np.nan_to_num(elevation_array, nan=0.0)

    if params.drop_sea_level:
        from scipy import signal

        dropped_sea_level = (
            -(params.scale * params.sea_level_drop / 1000) / params.exaggeration
        )
        # print("dropped_sea_level", dropped_sea_level)
        elevation_array = np.where(
            np.abs(elevation_array) <= params.sea_level,
            dropped_sea_level,
            elevation_array,
        )

        # # More complicated algorithm that smooths shoreline. Not needed.
        # sea = np.abs(elevation_array) <= params.sea_level
        # kernel = np.ones((3, 3), dtype=np.int8)
        # kernel[1, 1] = 0
        # N = signal.convolve(sea, kernel, mode="same")
        # T = signal.convolve(np.ones_like(elevation_array), kernel, mode="same")
        # elevation_array = np.where(
        #     np.abs(elevation_array) <= params.sea_level,
        #     dropped_sea_level * (N / T),
        #     elevation_array,
        # )

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

        tri = (
            surface[x, y],
            (surface[x, y][0], surface[x, y][1], bot_height),
            (surface[x + 1, y][0], surface[x + 1, y][1], bot_height),
        )

        model.add_face(tri)
        tri = (
            surface[x, y],
            (surface[x + 1, y][0], surface[x + 1, y][1], bot_height),
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
            (surface[x, y][0], surface[x, y][1], bot_height),
            (surface[x + 1, y][0], surface[x + 1, y][1], bot_height),
        )

        model.add_face(tri)
        tri = (
            surface[x, y],
            (surface[x + 1, y][0], surface[x + 1, y][1], bot_height),
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

    # # Bot surface
    #    surface[:, :, 2] = bot_height
    #    for x in range(steps - 1):
    #        for y in range(steps - 1):
    #            if ((x + y) % 2) == 0:
    #                model.add_face(
    #                    [surface[x, y], surface[x + 1, y], surface[x + 1, y + 1]]
    #                )
    #                model.add_face(
    #                    [surface[x, y], surface[x + 1, y + 1], surface[x, y + 1]]
    #                )
    #            else:
    #                model.add_face([surface[x, y + 1], surface[x, y], surface[x + 1, y]])
    #                model.add_face(
    #                    [surface[x + 1, y], surface[x + 1, y + 1], surface[x, y + 1]]
    #                )

    return model


# FIXME: Needs to be simplified now no longer trying to be overly clever.
def triangulate_base(
    boundary: BBox,
    origin: LLA,
    params: STLParameters,
    steps: int,
) -> ezdxf.render.MeshBuilder:
    model = ezdxf.render.MeshBuilder()
    south, west, north, east = boundary
    steps = steps // 8

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

    # model.add_face([east_top[0], east_bot[0], east_top[-1]][::-1])
    # model.add_face([east_top[-1], east_bot[0], east_bot[-1]][::-1])

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

    # model.add_face([west_top[0], west_bot[0], west_top[-1]])
    # model.add_face([west_top[-1], west_bot[0], west_bot[-1]])

    # bot of base
    for i in range(steps - 1):
        model.add_face([north_bot[i], south_bot[i], north_bot[i + 1]][::-1])
        model.add_face([north_bot[i + 1], south_bot[i], south_bot[i + 1]][::-1])

    # top of base
    # for i in range(steps - 1):
    #     model.add_face([north_top[i], south_top[i], north_top[i + 1]])
    #     model.add_face([north_top[i + 1], south_top[i], south_top[i + 1]])

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

    if params.bottom_holes:
        # Corner bottom holes
        offset = params.bottom_hole_offset
        sides = params.bottom_hole_sides
        radius = params.bottom_hole_padding + params.bottom_hole_diameter / 2
        depth = params.bottom_hole_depth
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


def lla_to_model(
    lat_lon_alt: LLA, origin_lat_lon_alt: LLA, params: STLParameters
) -> ENU:
    """
    Convert latitude, longitude, and altitude (LLA) coordinates
    to model ENU Cartesian coordinates in millimeters
    """

    lat, lon, alt = lat_lon_alt
    origin_lat, origin_lon, origin_alt = origin_lat_lon_alt

    east, north = lambert_conformal_conic(lat, lon, center_meridian=origin_lon)
    center_east, center_north = lambert_conformal_conic(
        origin_lat, origin_lon, center_meridian=origin_lon
    )

    east = east - center_east
    north = north - center_north
    alt = alt - origin_alt

    up = alt * params.exaggeration
    enu_scaled = np.asarray([east, north, up])
    enu_scaled /= params.scale
    enu_scaled *= 1000  # meters to mm

    return (enu_scaled[0], enu_scaled[1], enu_scaled[2])

    # else:
    #     assert False


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


# FIXME: Remove
# This routine no longer necessary now using projection. Should be removed.
def find_point_on_line(p1, p2, z3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    if z2 == z1:  # Avoid division by zero
        raise ValueError("Points p1 and p2 have the same z coordinate.")

    x3 = x1 + (x2 - x1) * ((z3 - z1) / (z2 - z1))
    y3 = y1 + (y2 - y1) * ((z3 - z1) / (z2 - z1))

    return x3, y3, z3


def lambert_conformal_conic(
    lat: float,
    lon: float,
    standard_parallel1: float = 33.0,
    standard_parallel2: float = 45.0,
    center_meridian: float = -96.0,
) -> Tuple[float, float]:
    """
    Convert latitude and longitude to Lambert Conformal Conic projection coordinates.

    :param lat: Latitude in degrees.
    :param lon: Longitude in degrees.
    :param standard_parallel1: First standard parallel.
    :param standard_parallel2: Second standard parallel.
    :param center_meridian: Longitude of the central meridian.
    :return: (x, y) coordinates in the Lambert Conformal Conic projection.
    """

    # Convert degrees to radians
    lat = math.radians(lat)
    lon = math.radians(lon)
    standard_parallel1 = math.radians(standard_parallel1)
    standard_parallel2 = math.radians(standard_parallel2)
    center_meridian = math.radians(center_meridian)

    # Ellipsoid parameters for WGS 84
    a = 6378137  # semi-major axis
    f = 1 / 298.257223563  # flattening
    e = math.sqrt(f * (2 - f))  # eccentricity

    # Calculate the scale factor at the standard parallels
    m1 = math.cos(standard_parallel1) / math.sqrt(
        1 - e**2 * math.sin(standard_parallel1) ** 2
    )
    m2 = math.cos(standard_parallel2) / math.sqrt(
        1 - e**2 * math.sin(standard_parallel2) ** 2
    )
    t = math.tan(math.pi / 4 - lat / 2) / (
        (1 - e * math.sin(lat)) / (1 + e * math.sin(lat))
    ) ** (e / 2)
    t1 = math.tan(math.pi / 4 - standard_parallel1 / 2) / (
        (1 - e * math.sin(standard_parallel1)) / (1 + e * math.sin(standard_parallel1))
    ) ** (e / 2)
    t2 = math.tan(math.pi / 4 - standard_parallel2 / 2) / (
        (1 - e * math.sin(standard_parallel2)) / (1 + e * math.sin(standard_parallel2))
    ) ** (e / 2)

    # Calculate the scale factor n
    n = math.log(m1 / m2) / math.log(t1 / t2)

    # Calculate the projection constants F and rho0
    F = m1 / (n * t1**n)
    rho = a * F * t**n
    rho0 = a * F * t1**n

    # Calculate the projected coordinates
    x = rho * math.sin(n * (lon - center_meridian))
    y = rho0 - rho * math.cos(n * (lon - center_meridian))

    return x, y


if __name__ == "__main__":
    sys.exit(main())
