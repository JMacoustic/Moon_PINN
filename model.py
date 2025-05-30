import numpy as np
import triangle as tr
import matplotlib.pyplot as plt


def generate_mesh(outer_size=1.0, inner_size=0.5, hole_center=(0.0, 0.0), grid_size=(3, 3)):
    points = []
    segments = []
    holes = []

    tile_width = outer_size
    tile_height = outer_size

    point_offset = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            offset_x = i * tile_width
            offset_y = j * tile_height

            # Outer square (counter-clockwise)
            w_outer = outer_size / 2
            outer = [
                [offset_x - w_outer, offset_y - w_outer],
                [offset_x + w_outer, offset_y - w_outer],
                [offset_x + w_outer, offset_y + w_outer],
                [offset_x - w_outer, offset_y + w_outer]
            ]
            outer_segments = [[point_offset + k, point_offset + (k + 1) % 4] for k in range(4)]

            # Inner square (clockwise)
            w_inner = inner_size / 2
            cx, cy = hole_center
            cx += offset_x
            cy += offset_y
            inner = [
                [cx - w_inner, cy - w_inner],
                [cx + w_inner, cy - w_inner],
                [cx + w_inner, cy + w_inner],
                [cx - w_inner, cy + w_inner]
            ]
            inner_segments = [[point_offset + 4 + k, point_offset + 4 + (k + 1) % 4] for k in range(4)]

            points.extend(outer + inner)
            segments.extend(outer_segments + inner_segments)
            holes.append([cx, cy])
            point_offset += 8

    A = dict(vertices=np.array(points),
             segments=np.array(segments),
             holes=np.array(holes))

    try:
        B = tr.triangulate(A, 'p q')
        if 'vertices' not in B or 'triangles' not in B:
            raise RuntimeError("Triangulation failed or returned incomplete result")
    except Exception as e:
        print("Triangulation error:", e)
        return None

    if 'vertices' not in B or 'triangles' not in B:
        raise RuntimeError("Triangulation failed!")

    verts = B['vertices'].flatten().tolist()
    triIds = B['triangles'].flatten().tolist()

    # Build boundary edges
    edges_set = set()
    for tri in B['triangles']:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            edge = tuple(sorted((a, b)))
            edges_set.add(edge)

    edges_list = list(edges_set)

    verts_array = np.array(verts).reshape(-1, 2)
    y_max = np.max(verts_array[:, 1])
    tol = 1e-6
    bc_mask = np.abs(verts_array[:, 1] - y_max) < tol
    bc_ids = np.where(bc_mask)[0].tolist()

    return {
        'verts': verts,
        'triIds': triIds,
        'edgeIds': edges_list,
        'bcIds' : bc_ids
    }


# Test and plot
if __name__ == "__main__":
    mesh = generate_mesh()
    vertices = np.array(mesh['verts']).reshape(-1, 2)
    triangles = np.array(mesh['triIds']).reshape(-1, 3)
    bc_ids = mesh['bcIds']

    plt.figure(figsize=(6, 6))
    for tri in triangles:
        pts = vertices[tri]
        plt.fill(*zip(*pts), edgecolor='black', fill=False)

    plt.plot(vertices[:, 0], vertices[:, 1], 'ko', markersize=2, label="All vertices")
    plt.plot(vertices[bc_ids, 0], vertices[bc_ids, 1], 'ro', markersize=4, label="BC nodes")

    plt.axis('equal')
    plt.legend()
    plt.title('Square Plate with Square Hole - Triangulation')
    plt.show()