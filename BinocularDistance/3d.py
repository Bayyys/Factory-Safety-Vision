# coding=utf-8

import numpy as np
from argparse import Namespace

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

camera_matrix = {'xc': 127.5, 'zc': 127.5, 'f': 128}
camera_matrix = Namespace(**camera_matrix)


def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """

    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for i in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * \
        Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * \
        Y[::scale, ::scale] / camera_matrix.f
    #

    XYZ = np.concatenate((X[..., np.newaxis], Y[::scale, ::scale]
                         [..., np.newaxis], Z[..., np.newaxis]), axis=X.ndim)
    return XYZ


image = Image.open("disp450.bmp")
image = image.resize((256, 256))
img = np.asarray(image)

XYZ = get_point_cloud_from_z(img, camera_matrix, scale=1)

ax = plt.figure(1).gca(projection='3d')
ax.plot(np.ndarray.flatten(XYZ[::, ::, 0]), np.ndarray.flatten(
    XYZ[::, ::, 1]), np.ndarray.flatten(XYZ[::, ::, 2]), 'b.', markersize=0.2)

plt.title('point cloud')
plt.show()
