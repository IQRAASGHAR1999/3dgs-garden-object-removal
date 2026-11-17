import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("GS_REPO", Path(__file__).resolve().parent.parent / "gaussian-splatting"))
sys.path.insert(0, str(REPO))

from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import getWorld2View2


def load_cameras(source, images="images_4"):
    info = readColmapSceneInfo(str(source), images, "", True, False)
    return list(info.train_cameras) + list(info.test_cameras)


def camera_matrices(cam, width, height):
    w2c = getWorld2View2(cam.R, cam.T).astype(np.float64)
    fx = width / (2.0 * np.tan(cam.FovX / 2.0))
    fy = height / (2.0 * np.tan(cam.FovY / 2.0))
    
    K = np.array([[fx, 0.0, (width - 1) / 2.0],
                  [0.0, fy, (height - 1) / 2.0],
                  [0.0, 0.0, 1.0]])
    return w2c, K


def project(xyz, w2c, K):
    pc = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-8, 1e-8, z)
    u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
    return u, v, z


def camera_center_and_axis(w2c):
    c2w = np.linalg.inv(w2c)
    return c2w[:3, 3], c2w[:3, 2]


def closest_point_to_axes(centers, axes):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centers, axes):
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)