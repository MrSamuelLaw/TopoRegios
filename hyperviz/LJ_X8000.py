from numba import jit, njit, prange
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
from time import time as now
from asyncio import create_task

"""
AUTHOR: Samuel Law
Date: 2/3/2022
Description: Module for importing scan data from and LJ_X8000 bmp
"""

# runtime constants for an LJ_X8000
BIT_RESOLUTION = 2**15  # bits
UPPER_LIMIT = 220   # mm
LOWER_LIMIT = -220  # mm
HEIGHT_DISPLACEMENT_RESOLUTION = (UPPER_LIMIT-LOWER_LIMIT)/BIT_RESOLUTION  # mm/bit
ZERO_POINT = BIT_RESOLUTION/2  # bit_res/2 because zero is in the middle of upper and lower limit.
EPS = 1      # mm
DX =  0.100  # mm
DYDX = 1     # mm


def bmp_to_point_cloud(path: Union[Path, str], extra_offset=0.0) -> Tuple[o3d.geometry.PointCloud, str]:
    """Takes a bmp file path and converts
    it to an open3d point cloud."""
    t1 = now()
    data = np.array(Image.open(str(path)), dtype=np.int16)
    t2 = now()
    data = bmp_data_to_height_map(data)
    t3 = now()
    data = height_map_to_np_array(data, extra_offset=extra_offset)
    t4 = now()
    point_cloud = np_array_to_point_cloud(data)
    t5 = now()
    estimate_normals(point_cloud)
    t6 = now()
    lines = [
        f'read duration: {t2 - t1:.5f} sec',
        f'height map computation duration: {t3 - t2:.5f} sec',
        f'height map -> numpy 3D point array duration: {t4 - t3:.5f} sec',
        f'numpy 3D point array -> O3DPointCloud duration: {t5 - t4:.5f} sec',
        f'O3DPointCloud Estimate Normals duration: {t6 - t5}: sec',
        f'total time = {t6 - t1:.5f}: sec'
    ]
    text = '\n'.join(lines)
    return point_cloud, text


def bmp_to_np_array(path: Union[Path, str], extra_offset=0.0) -> np.array:
    data = np.array(Image.open(str(path)))
    data = bmp_data_to_height_map(data)
    data = height_map_to_np_array(data, extra_offset=extra_offset)
    return data


@njit()
def get_height(pixel):
    """Formula provided by Keyence to convert pixels to height.
    The 0.01343 and -220 come from the sensor settings."""
    R, G, B = pixel
    decimal = int((G << 7) | ((R & 0x07) << 4) | (B & 0x0f))
    return (decimal - ZERO_POINT)*HEIGHT_DISPLACEMENT_RESOLUTION


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
    return point_cloud


# this function is the slowest function in the point cloud conversion process
# if this function can be sped up, loading in point clouds will feel nearly instantanious
def estimate_normals(pointcloud) -> o3d.geometry.PointCloud:
    pointcloud.estimate_normals()