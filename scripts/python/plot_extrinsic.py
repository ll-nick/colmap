# Quick script to visually validate extrinsics extracted using extract_calib.py
# Author: Nick Le Large

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import collections
import argparse
import sys
from transforms3d import affines, quaternions

Transformation = collections.namedtuple(
    "Transformation", ["id", "qvec", "tvec", "camera_id", "name"])

def inverse_transformation(tf):
    R = tf[:3, :3]
    R_inv = np.transpose(R)
    t = tf[:3, 3]
    t_inv = -np.matmul(R_inv,t)
    tf_inv = affines.compose(t_inv, R_inv, np.ones(3))

    return tf_inv

def read_extrinsic(path):
    transformations = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                transformations[image_id] = Transformation(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name)
    return transformations

def draw_cos(ax, tf, axislength = 0.2, text = ''):
    # Definitions in local coordinate frame
    origin_loc = [0, 0, 0, 1]
    x_loc = [axislength, 0, 0, 1]
    y_loc = [0, axislength, 0, 1]
    z_loc = [0, 0, axislength, 1]
    text_pos_loc = [0, 0, -0.1, 1]

    # Transform using given tf
    origin_glob = np.matmul(tf, origin_loc)
    x_glob = np.matmul(tf, x_loc)
    y_glob = np.matmul(tf, y_loc)
    z_glob = np.matmul(tf, z_loc)
    text_pos_glob = np.matmul(tf, text_pos_loc)
    
    # Plot axes
    ax.plot([origin_glob[0], x_glob[0]], [origin_glob[1],x_glob[1]], [origin_glob[2],x_glob[2]], color='r')
    ax.plot([origin_glob[0], y_glob[0]], [origin_glob[1],y_glob[1]], [origin_glob[2],y_glob[2]], color='g')
    ax.plot([origin_glob[0], z_glob[0]], [origin_glob[1],z_glob[1]], [origin_glob[2],z_glob[2]], color='b')

    # Add text
    ax.text(text_pos_glob[0], text_pos_glob[1], text_pos_glob[2], text)

def fix_axes(ax, coord_range):
    # Hacky way of fixing axes ranges by plotting invisible cube with maximum range
    Xmin = coord_range[0,0]
    Xmax = coord_range[0,1]
    Ymin = coord_range[1,0]
    Ymax = coord_range[1,1]
    Zmin = coord_range[2,0]
    Zmax = coord_range[2,1]

    max_range = np.array([Xmax-Xmin, Ymax-Ymin, Zmax-Zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(Xmax+Xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Ymax+Ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Zmax+Zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

def plot_extrinsic(extrinsic, axis_length):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    coord_range = np.block([[np.ones((3,1)) * sys.float_info.max, np.ones((3,1)) * sys.float_info.min]])
    for _, e in extrinsic.items():
        t = e.tvec        
        R = quaternions.quat2mat(e.qvec)
        tf = inverse_transformation(affines.compose(t, R, np.ones(3)))
        draw_cos(ax, tf, axislength = axis_length, text = '')

        for idx, coord in enumerate(t):
            if coord < coord_range[idx, 0]:
                coord_range[idx, 0] = coord
            if coord > coord_range[idx, 1]:
                coord_range[idx, 1] = coord

    fix_axes(ax, coord_range)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and write COLMAP binary and text models")
    parser.add_argument("extrinsic_calib", type=str, help="path to textfile containing extrinsic calibration.")
    parser.add_argument("--axis-length", type=float, required=False, default=0.2, help="Length of axis unit vectors in visualization.")
    args = parser.parse_args()

    extrinsic = read_extrinsic(path=args.extrinsic_calib)
    plot_extrinsic(extrinsic, args.axis_length)
