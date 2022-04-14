from numba import njit, prange
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
from typing import Union

"""
AUTHOR: Samuel Law
Date: 2/3/2022
Description: Module for importing scan data from and LJ_X8000 bmp
"""

# runtime constants for an LJ_X8000
HEIGHT_DISPLACEMENT_RESOLUTION =  0.01343  # mm
LOWER_LIMIT = -220  # mm
EPS = 1      # mm
DX =  0.100  # mm
DYDX = 1     # mm


def bmp_to_point_cloud(path: Union[Path, str], extra_offset=0.0) -> o3d.geometry.PointCloud:
    """Takes a bmp file path and converts
    it to an open3d point cloud."""
    data = np.array(Image.open(str(path)))
    data = bmp_data_to_height_map(data)
    data = height_map_to_np_array(data, extra_offset=extra_offset)
    point_cloud = np_array_to_point_cloud(data)
    return point_cloud


def bmp_to_np_array(path: Union[Path, str], extra_offset=0.0) -> np.array:
    data = np.array(Image.open(str(path)))
    data = bmp_data_to_height_map(data)
    data = height_map_to_np_array(data, extra_offset=extra_offset)
    return data


def point_cloud_to_ply(path, point_cloud):
    path = Path(path)
    name = path.name
    index = name.rfind('.')
    if index > -1:
        name = name[:index] + '.ply'
    else:
        name = name + '.ply'
    path = path.parent.joinpath(name)
    o3d.io.write_point_cloud(str(path), point_cloud)


@njit()
def get_height(pixel):
    """Formula provided by Keyence to convert pixels to height.
    The 0.01343 and -220 come from the sensor settings."""
    R, G, B = pixel
    return HEIGHT_DISPLACEMENT_RESOLUTION*int((G << 7) | (R & (0x07) << 4) | (B & 0x0f)) + LOWER_LIMIT


@njit(parallel=True)
def bmp_data_to_height_map(bmp_data: np.array) -> np.array:
    """Formula that leverages GPU capabilities to
    convert BPM data to a heigt map"""
    # convert the array to a vector of RGB vectors
    length, width, depth = bmp_data.shape
    rgb_array = bmp_data.reshape(length*width, depth)
    # create thread safe memory block
    height_data = np.zeros(length*width, dtype=np.float64)
    # loop over using GPU hardware accelerations
    for index in prange(length*width):
        height_data[index] = get_height(rgb_array[index])
    # return a 2D array where each value is the height
    return height_data.reshape(length, width)


def height_map_to_np_array(height_data: np.array, extra_offset=0.0) -> o3d.geometry.PointCloud:
    """Applys the xy parameters to the data
    to return a point cloud"""
    # grab the index where there is data
    rows, cols = np.where(height_data > (LOWER_LIMIT + EPS + extra_offset))
    #convert x, y, h data to rows in an array
    points = np.column_stack((rows*DX, cols*(DX*DYDX), height_data[rows, cols]))
    return points
    

def np_array_to_point_cloud(np_array: np.array) -> o3d.geometry.PointCloud:
    # convert to point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np_array)
    point_cloud.estimate_normals()
    return point_cloud

