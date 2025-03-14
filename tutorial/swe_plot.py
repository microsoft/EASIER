# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pyvista as pv
import imageio.v2 as imageio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--filename", type=str)
args = parser.parse_args()

frames = []
for i in range(100):
    data = np.load(f"{args.data_dir}/data{i:03d}.npz")
    points = np.concatenate(
        (data['x'], data['y'], data['z']*2), axis=0).transpose()
    point_cloud = pv.PolyData(points)
    surface = point_cloud.delaunay_2d().extract_geometry()

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(surface, color='skyblue', smooth_shading=True)
    plotter.render()
    frame = plotter.screenshot()
    frames.append(frame)
    plotter.clear()

gif_filename = args.gif_filename
imageio.mimwrite(gif_filename, frames, loop=0, duration=0.2)
