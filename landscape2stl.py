#!/usr/bin/env python

# Copyright 2023, Gavin E. Crooks and contributors
#
# This source code is licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.
#
#
# llandscape2stl: High resolution terrain models for 3D printing
#
# See README for more information
#
#
# Gavin E. Crooks 2023
#


# Note:
# latitude is north-south
# longitude is west-east (long way around)

import argparse
import math
import os
import sys
from math import pi
from os import path
from typing import Any

import numpy as np
import py3dep
import xarray as xr
from stl import mesh  # from package numpy-stl
from typing_extensions import TypeAlias

# Many units and coordinate systems. Use various TypeAlias in desperate effort to keep everything straight
LLA: TypeAlias = Any  # latitude, longitude, altitude (in meters) coordinates
ENU: TypeAlias = Any  # East, North, Up model coordinates (millimeters)
MM: TypeAlias = float  # millimeters
Meters: TypeAlias = float  # meters
Degrees: TypeAlias = float

# Various presets featuring varied terrains for testing and development.
presets = {
    "half_dome": [(37.76, -119.57, 37.72, -119.52), 24_000],  # 37.75°N 119.53
    "west_of_half_dome": [(37.76, -119.62, 37.72, -119.57), 24_000],  # 37.75°N 119.53
    "whitney": [(36.60, -118.33, 36.56, -118.28), 24_000],  # High point test
    "yosemite_west": [(37.80, -119.80, 37.70, -119.65), 62_500],
    "yosemite_valley": [(37.80, -119.65, 37.70, -119.50), 62_500],
    "clouds_rest": [(37.80, -119.50, 37.70, -119.35), 62_500],
    "grand_canyon": [(36.2000, -112.2000, 36.0000, -111.950), 125_000],
    "shasta": [(41.5000, -122.3500, 41.3000, -122.0500), 125_000],
    "shasta_west": [(41.5000, -122.6500, 41.3000, -122.3500), 125_000],
    "joshua_tree": [(34.0000, -116.0000, 33.8, -115.75), 125_000],
    "owens_valley": [(38, -120, 36.5, -118), 1_000_000],
    "denali": [(63.5000, -152.0000, 62.5000, -150.0000), 1_000_000],
}


standard_scales = [
    24_000,  # 1" = 2000', about 2.5" to 1 mile
    62_500,  # about 1" to 1 mile
    125_000,  # about 1" to 2 miles
    1_000_000,  # about 1" to 16 miles
]


default_cache = "cache"
default_scale = 62_500
default_steps = 1024  # change to 16 for faster debugging can be helpful
default_resolution: Meters = 10
meters_to_mm = 1000.0  # TODO: mm_per_meter
default_base: Meters = -100.0  # meters Lowest point in US is -86 m
default_exaggeration = 1.0
default_magnets: Degrees = 0.05
default_sea_level: Meters = 1.0  # surprisingly works better than 0 meters


default_scale_magnets = {
    24_000: 0.01,
    62_500: 0.025,
    125_000: 0.05,
    1_000_000: 0.25,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create landscape STLs")
    parser.add_argument(
        "coordinates",
        metavar="N W S E",
        type=float,
        nargs="*",
        help="Latitude/longitude coordinates for slab (Order North edge, west edge, south edge, east edge)",
    )

    parser.add_argument("--preset", dest="preset", choices=presets.keys())

    parser.add_argument("--scale", dest="scale", type=float, help="Map scale")

    parser.add_argument(
        "--exaggeration",
        dest="exaggeration",
        type=float,
        default=1.0,
        help="Vertical exaggeration",
    )

    parser.add_argument(
        "--magnets", dest="magnets", type=float, help="Magnet spacing (in degrees)"
    )

    parser.add_argument("--name", dest="name", type=str, help="Filename for model")

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

    if not args["coordinates"]:
        parser.print_help()
        return 0

    if args["magnets"] is None:
        args["magnets"] = default_scale_magnets[args["scale"]]

    if args["name"]:
        name = args["name"]

    create_stl(
        args["coordinates"],
        args["scale"],
        args["exaggeration"],
        magnets=args["magnets"],
        name=name,
    )

    return 0


# todo: rename coords to tile_coords?


def create_stl(
    coords,
    scale=default_scale,
    exaggeration=default_exaggeration,
    steps=default_steps,
    resolution=default_resolution,
    magnets=default_magnets,
    name=None,
):
    origin = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2, -100000

    # adjust origin so that 4 corners are equal height
    north, west, south, east = coords
    alt = 0.0
    west_north = lla_to_ecef([north, west, alt])
    east_north = lla_to_ecef([north, east, alt])
    east_south = lla_to_ecef([south, east, alt])
    west_south = lla_to_ecef([south, west, alt])
    new_origin_ecef = [
        (west_north[0] + east_south[0] + west_south[0] + east_north[0]) / 4,
        (west_north[1] + east_south[1] + west_south[1] + east_north[1]) / 4,
        (west_north[2] + east_south[2] + west_south[2] + east_north[2]) / 4,
    ]
    new_origin = ecef_to_lla(new_origin_ecef)

    # Honestly, dunno why this works.
    origin = origin[0] * 2 - new_origin[0], new_origin[1], 0.0
    crns = corners_to_model(coords, -0, origin, scale, exaggeration)

    elevation = download_elevation(coords, steps, resolution)

    print("Building terrain...")
    surface = elevation_to_surface(elevation, origin, scale, exaggeration)

    # FIXME Ocean
    # surface = np.where(surface==0.0, surface-100, surface)
    print("Triangulating...")
    triangles = triangulate_surface(surface, coords, origin, scale, exaggeration)

    base_triangles = triangulate_base(coords, origin, scale, exaggeration, magnets)

    triangles = np.concatenate([triangles, base_triangles])

    min_height = np.min(triangles)

    if name is None:
        filename = "{:.2f}_{:.2f}_{:.2f}_{:.2f}.stl".format(*coords)
    else:
        filename = name + ".stl"

    data = np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype)
    data["vectors"] = triangles
    the_mesh = mesh.Mesh(data.copy())

    print(f"Saving {filename}")
    the_mesh.save(filename)

    print()


def triangle_normal(A, B, C):
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)

    AB = B - A
    AC = C - A

    normal = np.cross(AB, AC)
    normal_normalized = normal / np.linalg.norm(normal)

    return normal_normalized


def lla_to_ecef(lat_lon_alt: LLA):
    """Convert latitude, longitude, altitude (LLA) coordinates to Earth-Centered, Earth-Fixed (ECEF) Cartesian
    coordinates"""
    latitude, longitude, altitude = lat_lon_alt

    # Constants for WGS84
    a = 6378137.0  # Equatorial radius (meters)
    e = 0.08181919084  # Eccentricity
    # e = 0 # Approximate earth as sphere

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


def ecef_to_lla(ecef):
    x, y, z = ecef

    # Constants for WGS84
    a = 6378137.0  # Semi-major axis
    e_sq = 0.00669437999014  # Square of eccentricity
    # e_sq = 0.0

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


def lla_to_enu(lat, lon, alt, origin_lat, origin_lon, origin_alt):
    """Convert latitude, longitude, and altitude (LLA) coordinates to local Cartesian coordinates
    (East-North-Up or ENU) relative to a given origin point."""
    # Convert origin to ECEF coordinates
    x_target, y_target, z_target = lla_to_ecef([lat, lon, alt])
    x_origin, y_origin, z_origin = lla_to_ecef([origin_lat, origin_lon, origin_alt])

    # Calculate ECEF vector between origin and target point
    dx, dy, dz = x_target - x_origin, y_target - y_origin, z_target - z_origin
    # print(dx, dy, dz)

    # Convert origin latitude and longitude to radians
    lat_rad = math.radians(origin_lat)
    lon_rad = math.radians(origin_lon)

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

    return enu[0], enu[1], enu[2]


def lla_to_model(
    lat, lon, alt, origin_lat, origin_lon, origin_alt, scale, exaggeration
):
    """Convert latitude, longitude, and altitude (LLA) coordinates to model ENU Cartesian coordinates (millimeters)"""
    enu = lla_to_enu(lat, lon, alt * exaggeration, origin_lat, origin_lon, origin_alt)
    enu = np.asarray(enu)
    enu /= scale
    # enu *= (1.0, 1.0, exaggeration)
    enu *= meters_to_mm

    return enu


def download_elevation(coords, steps=default_steps, resolution=default_resolution):
    north, west, south, east = coords

    xcoords = np.linspace(west, east, steps)
    ycoords = np.linspace(south, north, steps)

    filename = "{:.2f}_{:.2f}_{:.2f}_{:.2f}.nc".format(*coords)

    if not os.path.exists(default_cache):
        os.mkdir(default_cache)

    fname = os.path.join(default_cache, filename)

    if not os.path.exists(fname):
        print("Downloading elevation data... ", end="", flush=True)
        elevation = py3dep.elevation_bygrid(
            xcoords, ycoords, crs="EPSG:4326", resolution=10
        )  # or 30
        elevation.to_netcdf(fname)
        print("Done", flush=True)

    print("Loading elevation data from cache...", end="", flush=True)
    elevation = xr.open_dataset(fname)
    print("", flush=True)

    return elevation


def elevation_to_surface(elevation, origin, scale, exaggeration):
    ycoords = np.asarray(elevation.coords["y"])
    xcoords = np.asarray(elevation.coords["x"])
    steps = len(ycoords)
    elevation = np.asarray(elevation.to_array()).reshape((steps, steps)).T
    elevation = np.nan_to_num(elevation, nan=0.0)

    # Uncomment this next line to drop the elevation of ocean
    # elevation[elevation <= default_sea_level] = -100

    surface = np.zeros(shape=(steps, steps, 3))

    for x in range(steps):
        for y in range(steps):
            lat = ycoords[y]
            lon = xcoords[x]
            alt = elevation[x, y]
            surface[x, y] = lla_to_model(lat, lon, alt, *origin, scale, exaggeration)

    return surface


def normalize(v):
    return v / np.linalg.norm(v)


def angle_between(v, w):
    return np.arccos(np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)))


def rotation_matrix(axis, theta):
    kx, ky, kz = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c

    return np.array(
        [
            [kx * kx * C + c, kx * ky * C - kz * s, kx * kz * C + ky * s],
            [kx * ky * C + kz * s, ky * ky * C + c, ky * kz * C - kx * s],
            [kx * kz * C - ky * s, ky * kz * C + kx * s, kz * kz * C + c],
        ]
    )


def rotate_vector(v, w):
    v_normalized = normalize(v)
    w_normalized = normalize(w)

    axis = np.cross(v_normalized, w_normalized)
    axis_normalized = normalize(axis)

    theta = angle_between(v_normalized, w_normalized)

    R = rotation_matrix(axis_normalized, theta)

    return R


def corners_to_model(coords, alt, origin, scale, exaggeration):
    north, west, south, east = coords
    west_north: ENU = lla_to_model(north, west, alt, *origin, scale, exaggeration)
    east_north: ENU = lla_to_model(north, east, alt, *origin, scale, exaggeration)
    east_south: ENU = lla_to_model(south, east, alt, *origin, scale, exaggeration)
    west_south: ENU = lla_to_model(south, west, alt, *origin, scale, exaggeration)

    return west_north, west_south, east_south, east_north


def triangulate_surface(surface, coords, origin, scale, exaggeration):
    triangles = []
    steps = surface.shape[0]
    north, west, south, east = coords

    # Top surface
    for x in range(steps - 1):
        for y in range(steps - 1):
            if ((x + y) % 2) == 0:
                triangles.append(
                    [surface[x, y], surface[x + 1, y], surface[x + 1, y + 1]]
                )
                triangles.append(
                    [surface[x, y], surface[x + 1, y + 1], surface[x, y + 1]]
                )
            else:
                triangles.append([surface[x, y + 1], surface[x, y], surface[x + 1, y]])
                triangles.append(
                    [surface[x + 1, y], surface[x + 1, y + 1], surface[x, y + 1]]
                )

    corners = corners_to_model(coords, default_base, origin, scale, exaggeration)
    west_north_bot, west_south_bot, east_south_bot, east_north_bot = corners
    bot_west = west_north_bot[0]
    bot_north = west_north_bot[1]
    bot_alt = west_north_bot[2]
    bot_east = east_south_bot[0]
    bot_south = east_south_bot[1]

    bot_height = bot_alt

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
        # curved edge
        # tri = (
        #     surface[x, y],
        #     find_point_on_line(surface[x, y], [surface[x, y][0], surface[x, y][1], default_base], bot_height),
        #     find_point_on_line(surface[x+1, y], [surface[x+1, y][0], surface[x+1, y][1], default_base], bot_height),
        #     # (xcoords[x], ycoords[x], bot_height),
        #     # (xcoords[x + 1], ycoords[x + 1], bot_height),
        # )
        triangles.append(tri)
        tri = (
            surface[x, y],
            (xcoords[x + 1], ycoords[x + 1], bot_height),
            surface[x + 1, y],
        )

        # Curved edge
        # tri = (
        #     surface[x, y],
        #     find_point_on_line(surface[x+1, y], [surface[x+1, y][0], surface[x+1, y][1], default_base], bot_height),
        #     # (xcoords[x + 1], ycoords[x + 1], bot_height),
        #     surface[x + 1, y],
        # )
        triangles.append(tri)

        #     mag_coords = find_point_on_line(coords0, coords1, mag_height)

    # north
    # print(west_north_bot, east_north_bot)
    xcoords = np.linspace(west_north_bot[0], east_north_bot[0], steps)
    ycoords = np.linspace(west_north_bot[1], east_north_bot[1], steps)

    for x in range(steps - 1):
        # print(surface[x, y])
        y = -1
        tri = (
            surface[x, y],
            (xcoords[x], ycoords[x], bot_height),
            (xcoords[x + 1], ycoords[x + 1], bot_height),
        )
        triangles.append(tri)
        tri = (
            surface[x, y],
            (xcoords[x + 1], ycoords[x + 1], bot_height),
            surface[x + 1, y],
        )
        triangles.append(tri)

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
        triangles.append(tri)
        tri = (
            surface[x, s],
            (xcoords[s + 1], ycoords[s + 1], bot_height),
            surface[x, s + 1],
        )
        triangles.append(tri)

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
        triangles.append(tri)
        tri = (
            surface[x, s],
            (xcoords[s + 1], ycoords[s + 1], bot_height),
            surface[x, s + 1],
        )
        triangles.append(tri)

    # bottom
    # triangles.append([west_south_bot, east_south_bot, west_north_bot])
    # triangles.append([east_north_bot, west_north_bot, east_south_bot])

    triangles = np.asarray(triangles)
    return triangles


# FIXME: Still not right?
# Magnet bounding rectangle constant altitude rather than z-height


def triangulate_base(coords, origin, scale, exaggeration, magnets):
    triangles = []
    north, west, south, east = coords

    default_base_height = 10  # mm
    base_alt = default_base - (
        default_base_height * scale / (meters_to_mm * exaggeration)
    )

    magnets_alt = (base_alt + default_base) / 2

    top_corners = corners_to_model(coords, default_base, origin, scale, exaggeration)
    west_north_top, west_south_top, east_south_top, east_north_top = top_corners

    bot_corners = corners_to_model(coords, base_alt, origin, scale, exaggeration)
    west_north_bot, west_south_bot, east_south_bot, east_north_bot = bot_corners

    mag_corners = corners_to_model(coords, magnets_alt, origin, scale, exaggeration)
    west_north_mag, west_south_mag, east_south_mag, east_north_mag = mag_corners

    # Add base

    # bot of base
    triangles.append([west_south_bot, east_south_bot, west_north_bot])
    triangles.append([east_north_bot, west_north_bot, east_south_bot])

    # top of base
    # triangles.append([west_south_top, east_south_top, west_north_top])
    # triangles.append([east_north_top, west_north_top, east_south_top])

    bot = west_north_bot[2]
    top = west_north_top[2]
    mid = west_north_mag[2]

    if magnets == 0.0:
        # if True:
        # Base sides (No magnets)
        triangles.append([east_south_bot, east_south_top, west_south_bot])
        triangles.append([west_south_top, west_south_bot, east_south_top])

        triangles.append([east_south_top, east_south_bot, east_north_bot])
        triangles.append([east_north_bot, east_north_top, east_south_top])

        triangles.append([east_north_bot, east_north_top, west_north_bot])
        triangles.append([west_north_top, west_north_bot, east_north_top])

        triangles.append([west_south_bot, west_south_top, west_north_bot])
        triangles.append([west_north_top, west_north_bot, west_south_top])

        triangles = np.asarray(triangles)
        return triangles

    def trianglate_hole(
        mag_coords, mag_normal, scale, exaggeration, mheight=8, mwidth=8
    ):
        mt = []
        mheight = 8
        mwidth = 8
        R = rotate_vector((0, 0, 1), mag_normal)
        mag_triangles, mag_square = magnet_hole(height=mheight, width=mwidth)
        for tri in mag_triangles:
            A, B, C = tri
            a = R.dot(A) + mag_coords
            b = R.dot(B) + mag_coords
            c = R.dot(C) + mag_coords

            mt.append([a, b, c])

        N, W, S, E = mag_square
        n = R.dot(N) + mag_coords
        w = R.dot(W) + mag_coords
        s = R.dot(S) + mag_coords
        e = R.dot(E) + mag_coords

        return mt, (n, w, s, e)

    mag_longs = np.arange(west + magnets / 2, east, magnets)
    mag_lats = np.arange(south + magnets / 2, north, magnets)

    long_steps = 1 + round((east - west) / magnets) * 2
    mag2_longs = np.linspace(west, east, long_steps)
    lat_steps = 1 + round((north - south) / magnets) * 2
    mag2_lats = np.linspace(south, north, lat_steps)
    # print(mag2_lats.shape)

    # print(mag2_lats)

    mag_alt = (base_alt + default_base) / 2
    mag_corners = corners_to_model(coords, mag_alt, origin, scale, exaggeration)
    mag_height = mag_corners[0][2]

    # # south

    mag_normal = triangle_normal(east_south_bot, east_south_top, west_south_bot)

    xcoords_top = np.linspace(west_south_top[0], east_south_top[0], long_steps)
    ycoords_top = np.linspace(west_south_top[1], east_south_top[1], long_steps)

    xcoords_bot = np.linspace(west_south_bot[0], east_south_bot[0], long_steps)
    ycoords_bot = np.linspace(west_south_bot[1], east_south_bot[1], long_steps)

    xcoords_mag = np.linspace(west_south_mag[0], east_south_mag[0], long_steps)
    ycoords_mag = np.linspace(west_south_mag[1], east_south_mag[1], long_steps)

    for i in range(1, long_steps, 2):
        mag_coords = [xcoords_mag[i], ycoords_mag[i], mid]
        tri, square = trianglate_hole(mag_coords, mag_normal, scale, exaggeration)
        triangles.extend(tri)

        mag_corners_coords = [
            (xcoords_top[i + 1], ycoords_top[i + 1], top),
            (xcoords_top[i - 1], ycoords_top[i - 1], top),
            (xcoords_bot[i - 1], ycoords_bot[i - 1], bot),
            (xcoords_bot[i + 1], ycoords_bot[i + 1], bot),
        ]
        for c in [-1, 0, 1, 2]:
            tri = square[c], square[c + 1], mag_corners_coords[c]
            triangles.append(tri)
            tri = square[c + 1], mag_corners_coords[c], mag_corners_coords[c + 1]
            triangles.append(tri)

    # # north

    mag_normal = triangle_normal(east_north_bot, west_north_bot, east_north_top)

    xcoords_top = np.linspace(west_north_top[0], east_north_top[0], long_steps)
    ycoords_top = np.linspace(west_north_top[1], east_north_top[1], long_steps)

    xcoords_bot = np.linspace(west_north_bot[0], east_north_bot[0], long_steps)
    ycoords_bot = np.linspace(west_north_bot[1], east_north_bot[1], long_steps)

    xcoords_mag = np.linspace(west_north_mag[0], east_north_mag[0], long_steps)
    ycoords_mag = np.linspace(west_north_mag[1], east_north_mag[1], long_steps)

    for i in range(1, long_steps, 2):
        mag_coords = [xcoords_mag[i], ycoords_mag[i], mid]
        tri, square = trianglate_hole(mag_coords, mag_normal, scale, exaggeration)
        triangles.extend(tri)

        mag_corners_coords = [
            (xcoords_top[i + 1], ycoords_top[i + 1], top),
            (xcoords_top[i - 1], ycoords_top[i - 1], top),
            (xcoords_bot[i - 1], ycoords_bot[i - 1], bot),
            (xcoords_bot[i + 1], ycoords_bot[i + 1], bot),
        ]
        mag_corners_coords = mag_corners_coords[::-1]
        for c in [-1, 0, 1, 2]:
            tri = square[c], square[c + 1], mag_corners_coords[c]
            triangles.append(tri)
            tri = square[c + 1], mag_corners_coords[c], mag_corners_coords[c + 1]
            triangles.append(tri)

    # west
    mag_normal = triangle_normal(west_south_bot, west_south_top, west_north_top)

    mag_lla0 = (mag_lats[0], west, mag_alt)
    mag_lla1 = (mag_lats[1], west, mag_alt)
    mwidth = (
        lla_to_model(*mag_lla1, *origin, scale, exaggeration)[1]
        - lla_to_model(*mag_lla0, *origin, scale, exaggeration)[1]
    )

    for i in range(1, lat_steps, 2):
        # print(i, mag2_lats[i])
        lat = mag2_lats[i]

        mag_lla0 = (lat, west, base_alt)
        mag_lla1 = (lat, west, default_base)
        coords0 = lla_to_model(*mag_lla0, *origin, scale, exaggeration)
        coords1 = lla_to_model(*mag_lla1, *origin, scale, exaggeration)
        mag_coords = find_point_on_line(coords0, coords1, mag_height)

        tri, square = trianglate_hole(
            mag_coords, mag_normal, scale, exaggeration, mwidth=9
        )
        triangles.extend(tri)

        xcoords_top = np.linspace(west_south_top[0], west_north_top[0], lat_steps)
        ycoords_top = np.linspace(west_south_top[1], west_north_top[1], lat_steps)

        xcoords_bot = np.linspace(west_south_bot[0], west_north_bot[0], lat_steps)
        ycoords_bot = np.linspace(west_south_bot[1], west_north_bot[1], lat_steps)

        mag_corners_coords = [
            (xcoords_top[i + 1], ycoords_top[i + 1], top),
            (xcoords_top[i - 1], ycoords_top[i - 1], top),
            (xcoords_bot[i - 1], ycoords_bot[i - 1], bot),
            (xcoords_bot[i + 1], ycoords_bot[i + 1], bot),
        ]
        mag_corners_coords = mag_corners_coords[::-1]
        for c in [-1, 0, 1, 2]:
            tri = square[c], square[c + 1], mag_corners_coords[c]
            triangles.append(tri)
            tri = square[c + 1], mag_corners_coords[c], mag_corners_coords[c + 1]
            triangles.append(tri)

    # east
    mag_normal = triangle_normal(east_south_top, east_south_bot, east_north_top)

    mag_lla0 = (mag_lats[0], east, mag_alt)
    mag_lla1 = (mag_lats[1], east, mag_alt)
    mwidth = (
        lla_to_model(*mag_lla1, *origin, scale, exaggeration)[1]
        - lla_to_model(*mag_lla0, *origin, scale, exaggeration)[1]
    )

    for i in range(1, lat_steps, 2):
        lat = mag2_lats[i]

        mag_lla0 = (lat, east, base_alt)
        mag_lla1 = (lat, east, default_base)
        coords0 = lla_to_model(*mag_lla0, *origin, scale, exaggeration)
        coords1 = lla_to_model(*mag_lla1, *origin, scale, exaggeration)
        mag_coords = find_point_on_line(coords0, coords1, mag_height)

        tri, square = trianglate_hole(mag_coords, mag_normal, scale, exaggeration)
        triangles.extend(tri)

        mag_corners = [
            (mag2_lats[i + 1], east, base_alt),
            (mag2_lats[i + 1], east, default_base),
            (mag2_lats[i - 1], east, default_base),
            (mag2_lats[i - 1], east, base_alt),
        ]
        mag_corners_coords = [
            lla_to_model(*lla, *origin, scale, exaggeration) for lla in mag_corners
        ]
        mag_corners_coords = mag_corners_coords[::1]
        for c in [-1, 0, 1, 2]:
            tri = square[c], square[c + 1], mag_corners_coords[c]
            triangles.append(tri)
            tri = square[c + 1], mag_corners_coords[c], mag_corners_coords[c + 1]
            triangles.append(tri)

    triangles = np.asarray(triangles)
    return triangles


def find_point_on_line(p1, p2, z3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    if z2 == z1:  # Avoid division by zero
        raise ValueError("Points p1 and p2 have the same z coordinate.")

    x3 = x1 + (x2 - x1) * ((z3 - z1) / (z2 - z1))
    y3 = y1 + (y2 - y1) * ((z3 - z1) / (z2 - z1))

    return x3, y3, z3


def magnet_hole(height=10, width=10):
    def PointsInCircum(r, n=100, z=0.0):
        return [
            (math.cos(2 * pi / n * x) * r, math.sin(2 * pi / n * x) * r, z)
            for x in range(0, n + 1)
        ]

    magnet_diameter = 6.00
    magnet_depth = 1.65
    magnet_padding = 0.25

    diameter = 6.5
    depth = 1.8
    sides = 8

    # hedge = 5 # half edge
    top_circle = np.asarray(PointsInCircum(diameter / 2, sides, 0.0))
    bot_circle = np.asarray(PointsInCircum(diameter / 2, sides, -depth))
    center_bot = (0.0, 0.0, -depth)

    triangles = []

    top_square = [
        [height / 2, 0, 0],
        [height / 2, width / 2, 0],
        [0, width / 2, 0],
        [-height / 2, width / 2, 0],
        [-height / 2, 0, 0],
        [-height / 2, -width / 2, 0],
        [0, -width / 2, 0],
        [height / 2, -width / 2, 0],
        [height / 2, 0, 0],
    ]

    circle = top_circle
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

    return np.asarray(triangles), (
        top_square[1],
        top_square[3],
        top_square[5],
        top_square[7],
    )


if __name__ == "__main__":
    sys.exit(main())
