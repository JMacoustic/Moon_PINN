from model import generate_mesh
from pinn import Finite_PINN, Offline, Online
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from time import perf_counter as default_timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mesh_data = generate_mesh()

# Setup PINN system
OFF = Offline(device=device, mesh_data=mesh_data, E=5.0, nu=0.3, t=1.0, rho=1.0, endtime=10.0, steps=100)
model = Finite_PINN().to(device)
ON = Online(OFF, model, device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
num_epochs = 1000

# Precompute (x, y, t) over entire mesh and time
x = OFF.X.view(-1, 1).to(device)
y = OFF.Y.view(-1, 1).to(device)
t = OFF.T.repeat(OFF.vertices.shape[0], 1).view(-1, 1).to(device)

loss_history = []
pde_history = []
bc_history = []
ic_history = []

for ep in range(num_epochs):
    model.train()
    t1 = default_timer()

    optimizer.zero_grad()

    u_x, u_y = model(x, y, t)
    u_pred = torch.cat((u_x, u_y), dim=1).view(OFF.T.shape[0], -1)

    pde = ON.PDE_loss(u_pred)
    bc = ON.BC_loss(u_pred)
    ic = ON.IC_loss(u_pred)
    loss = pde + 5 * bc + 2 * ic

    loss.backward()
    optimizer.step()

    t2 = default_timer()

    # Save losses
    loss_history.append(loss.item())
    pde_history.append(pde.item())
    bc_history.append(bc.item())
    ic_history.append(ic.item())

    print(f"Epoch:{ep+1}, IC:{ic:.2e}, BC:{bc:.2e}, PDE:{pde:.2e}")
# Inference & Visualization
model.eval()
with torch.no_grad():
    x_base = OFF.vertices[:, 0:1].to(device)
    y_base = OFF.vertices[:, 1:2].to(device)
    num_nodes = x_base.shape[0]
    num_steps = OFF.T.shape[0]

    frames = []
    for t in OFF.T:
        t_batch = t.repeat(num_nodes, 1)  # [num_nodes, 1]
        u_x, u_y = model(x_base, y_base, t_batch)
        frames.append((u_x.cpu().squeeze(), u_y.cpu().squeeze()))

# Base vertices (for rendering)
x_cpu = x_base.cpu().squeeze()
y_cpu = y_base.cpu().squeeze()
verts = torch.stack([x_cpu, y_cpu], dim=1)

# Edges from mesh
edge_indices = mesh_data['edgeIds']
edges = torch.stack([torch.stack([verts[i], verts[j]]) for i, j in edge_indices])

# Setup plot
fig, ax = plt.subplots(figsize=(6, 6))
lines = LineCollection(edges.numpy(), linewidths=0.5, colors='black')
ax.add_collection(lines)
ax.set_xlim(x_cpu.min()-1, x_cpu.max()+1)
ax.set_ylim(y_cpu.min()-1, y_cpu.max()+1)
ax.set_aspect('equal')
ax.axis('off')

# Update function
def update(frame_idx):
    ux, uy = frames[frame_idx]
    displaced = torch.stack([x_cpu + ux, y_cpu + uy], dim=1)
    new_edges = torch.stack([torch.stack([displaced[i], displaced[j]]) for i, j in edge_indices])
    lines.set_segments(new_edges.numpy())
    return lines,

# Animate
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=50, blit=True)
ani.save("pinn_motion_with_edges.gif", dpi=100, writer='pillow')
plt.show()


# plt.figure(figsize=(10, 6))
# plt.plot(loss_history, label='Total Loss')
# plt.plot(pde_history, label='PDE Loss')
# plt.plot(bc_history, label='BC Loss')
# plt.plot(ic_history, label='IC Loss')
# plt.yscale('log')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Evolution During Training')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("loss_curve.png", dpi=150)
# plt.show()