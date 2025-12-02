import numpy as np

class SoftBody2D:
    def __init__(self, triMesh, fixed_ids=[], edgeCompliance=100.0, areaCompliance=0.0):
        self.numParticles = len(triMesh['verts']) // 2
        self.numTris = len(triMesh['triIds']) // 3
        self.fixed_ids = fixed_ids

        self.pos = np.array(triMesh['verts'], dtype=np.float32)
        self.prevPos = np.copy(self.pos)
        self.vel = np.zeros(2 * self.numParticles, dtype=np.float32)

        self.triIds = triMesh['triIds']
        self.edgeIds = triMesh['triEdgeIds']
        self.restArea = np.zeros(self.numTris, dtype=np.float32)
        self.edgeLengths = np.zeros(len(self.edgeIds) // 2, dtype=np.float32)
        self.invMass = np.ones(self.numParticles, dtype=np.float32)

        for i in self.fixed_ids:
            self.invMass[i] = 0.0

        self.edgeCompliance = edgeCompliance
        self.areaCompliance = areaCompliance
        self.init_physics()

    def get_tri_area(self, i):
        id0, id1, id2 = self.triIds[3*i : 3*i+3]
        a, b, c = [self.pos[2*id:2*id+2] for id in (id0, id1, id2)]
        return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))

    def init_physics(self):
        for i in range(self.numTris):
            area = self.get_tri_area(i)
            self.restArea[i] = area
            pInvMass = 1.0 / (area / 3.0) if area > 0 else 0.0
            for j in range(3):
                pid = self.triIds[3*i + j]
                if self.invMass[pid] > 0:
                    self.invMass[pid] += pInvMass

        for i in range(len(self.edgeLengths)):
            id0, id1 = self.edgeIds[2*i], self.edgeIds[2*i+1]
            d = self.pos[2*id0:2*id0+2] - self.pos[2*id1:2*id1+2]
            self.edgeLengths[i] = np.linalg.norm(d)

    def pre_solve(self, dt, gravity):
        for i in range(self.numParticles):
            if self.invMass[i] == 0: continue
            self.vel[2*i:2*i+2] += gravity * dt
            self.prevPos[2*i:2*i+2] = self.pos[2*i:2*i+2].copy()
            self.pos[2*i:2*i+2] += self.vel[2*i:2*i+2] * dt
            if self.pos[2*i+1] < -4.0:  # floor
                self.pos[2*i+1] = 0.0
                self.vel[2*i+1] = 0.0

    def post_solve(self, dt):
        for i in range(self.numParticles):
            if self.invMass[i] == 0: continue
            self.vel[2*i:2*i+2] = (self.pos[2*i:2*i+2] - self.prevPos[2*i:2*i+2]) / dt

    def solve(self, dt):
        self.solve_edges(self.edgeCompliance, dt)
        self.solve_areas(self.areaCompliance, dt)

    def solve_edges(self, compliance, dt):
        alpha = compliance / (dt * dt)
        for i in range(len(self.edgeLengths)):
            id0, id1 = self.edgeIds[2*i], self.edgeIds[2*i+1]
            w0, w1 = self.invMass[id0], self.invMass[id1]
            w = w0 + w1
            if w == 0: continue
            d = self.pos[2*id0:2*id0+2] - self.pos[2*id1:2*id1+2]
            len_ = np.linalg.norm(d)
            if len_ == 0: continue
            grad = d / len_
            C = len_ - self.edgeLengths[i]
            s = -C / (w + alpha)
            self.pos[2*id0:2*id0+2] += grad * s * w0
            self.pos[2*id1:2*id1+2] -= grad * s * w1

    def solve_areas(self, compliance, dt):
        alpha = compliance / (dt * dt)
        for i in range(self.numTris):
            ids = self.triIds[3*i:3*i+3]
            a, b, c = [self.pos[2*id:2*id+2] for id in ids]
            signed_area = 0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
            area = abs(signed_area)
            C = area - self.restArea[i]
            grads = [
                np.array([b[1]-c[1], c[0]-b[0]]) * 0.5,
                np.array([c[1]-a[1], a[0]-c[0]]) * 0.5,
                np.array([a[1]-b[1], b[0]-a[0]]) * 0.5,
            ]
            w = sum(self.invMass[ids[j]] * np.dot(grads[j], grads[j]) for j in range(3))
            if w == 0.0:
                continue
            s = -C / (w + alpha)
            for j in range(3):
                self.pos[2*ids[j]:2*ids[j]+2] += grads[j] * s * self.invMass[ids[j]]

def generate_grid_mesh(nx, ny, dx=0.1, dy=0.1):
    verts = []
    triIds = []
    triEdgeIds = []
    id_map = lambda i, j: i * (nx + 1) + j

    for i in range(ny + 1):
        for j in range(nx + 1):
            verts.extend([j * dx, 1.0 - i * dy])

    for i in range(ny):
        for j in range(nx):
            v0 = id_map(i, j)
            v1 = id_map(i, j + 1)
            v2 = id_map(i + 1, j)
            v3 = id_map(i + 1, j + 1)
            triIds += [v0, v1, v2, v1, v3, v2]
            triEdgeIds += [v0, v1, v1, v2, v2, v0, v1, v3, v3, v2, v2, v1]

    return {
        'verts': verts,
        'triIds': triIds,
        'triEdgeIds': triEdgeIds
    }
