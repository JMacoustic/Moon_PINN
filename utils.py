import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

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