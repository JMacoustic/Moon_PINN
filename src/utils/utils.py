import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import json
import os
import random
import numpy as np
import torch

def triangulate(points):
    """
    Subdivide a 2D shape into triangles using Delaunay triangulation.
    
    Parameters:
        points (ndarray): Nx2 array of (x, y) coordinates.
        
    Returns:
        tri.simplices (ndarray): Mx3 array of indices of triangle vertices.
    """
    tri = Delaunay(points)
    return tri.simplices, tri

def plot_triangulation(points, triangles):
    plt.triplot(points[:,0], points[:,1], triangles)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.gca().set_aspect('equal')
    plt.title('Delaunay Triangulation')
    plt.show()


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For CUDA matmul determinism on recent CUDA: choose *one* of these values
    # (PyTorch recommends setting this when use_deterministic_algorithms is True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # This will throw if a nondeterministic op is used (good: it surfaces issues early)
        torch.use_deterministic_algorithms(True, warn_only=False)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def thickness_from_constraint(C: float, px: float, py: float, xoff: float) -> float:
    denom = (py + 0.5 * px + xoff)
    return max(1e-9, C / max(1e-9, denom))