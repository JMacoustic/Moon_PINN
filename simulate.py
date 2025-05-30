from model import generate_mesh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from XPBD import SoftBody2D

mesh = generate_mesh(outer_size=1.0, inner_size=0.5, hole_center=(0.0, 0.0), grid_size=(3, 3))
fixed_ids = [i for i, (x, y) in enumerate(np.reshape(mesh['verts'], (-1, 2)))
             if y > 2]  

body = SoftBody2D(mesh, fixed_ids=fixed_ids, edgeCompliance=1e-1, areaCompliance=1e-2)
gravity = np.array([0, -10], dtype=np.float32)
dt = 0.01

if mesh is None:
    raise RuntimeError("Mesh generation failed. Check geometry and triangulation.")
#--- Matplotlib Animation ---
fig, ax = plt.subplots()
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 4)
lines = [ax.plot([], [], 'bo-', lw=1)[0] for _ in range(len(mesh['triIds']) // 3)]

def update(frame):
    if np.any(np.isnan(body.pos)) or np.any(np.isinf(body.pos)):
        raise RuntimeError("Numerical instability detected: NaN or Inf in position.")
    
    body.pre_solve(dt, gravity)
    time = frame * dt
    amplitude = 0.1 
    frequency = 10.0  # Hz

    for i in fixed_ids:
        body.pos[2 * i + 1] += amplitude * np.sin(2 * np.pi * frequency * time)
        body.prevPos[2 * i+1] = body.pos[2 * i+1]

    for _ in range(10):  # XPBD iterations
        body.solve(dt)
    body.post_solve(dt)

    verts = body.pos.reshape(-1, 2)
    for i in range(0, len(mesh['triIds']), 3):
        tri = [verts[mesh['triIds'][i + j]] for j in range(3)] + [verts[mesh['triIds'][i]]]
        x, y = zip(*tri)
        lines[i // 3].set_data(x, y)
    return lines

ani = FuncAnimation(fig, update, frames=500, interval=30, blit=True)
plt.title("Deformable 2D Soft Body")
plt.show()