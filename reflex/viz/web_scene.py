"""Geometry and state helpers for the web viewer: MuJoCo model -> meshes a browser can draw.

The simulation stays in Python. The browser only needs, once, one mesh per body (in the
body's own frame) and then, many times a second, where each body is.

    meshes = body_meshes(model)          # {body_id: trimesh.Trimesh}
    state = pack_state(model, data)      # float32: time, then position and quaternion per body
    t, pos, quat = unpack_state(state)
"""
from __future__ import annotations

import mujoco
import numpy as np

VISUAL_GROUPS = (0, 1, 2)       # Menagerie keeps collision shapes in group 3
_DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)


def geom_rgba(model: mujoco.MjModel, g: int) -> np.ndarray:
    """Colour MuJoCo would draw the geom with: its own rgba if set, else its material's."""
    rgba, mat = model.geom_rgba[g], int(model.geom_matid[g])
    if mat >= 0 and np.allclose(rgba, _DEFAULT_RGBA):
        return model.mat_rgba[mat]
    return rgba


def _geom_mesh(model: mujoco.MjModel, g: int):
    import trimesh
    G = mujoco.mjtGeom
    kind, size = int(model.geom_type[g]), model.geom_size[g]
    if kind == G.mjGEOM_MESH:
        m = int(model.geom_dataid[g])
        v0, nv = int(model.mesh_vertadr[m]), int(model.mesh_vertnum[m])
        f0, nf = int(model.mesh_faceadr[m]), int(model.mesh_facenum[m])
        return trimesh.Trimesh(model.mesh_vert[v0:v0 + nv].copy(), model.mesh_face[f0:f0 + nf].copy(), process=False)
    if kind == G.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)
    if kind == G.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=float(size[0]))
    if kind == G.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=float(size[0]), height=2 * float(size[1]))
    if kind == G.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]))
    return None     # planes are drawn by the viewer as a grid


def body_meshes(model: mujoco.MjModel, groups=VISUAL_GROUPS) -> dict:
    """One coloured mesh per body, expressed in the body frame. Bodies with no shape are left out."""
    import trimesh
    parts: dict[int, list] = {}
    for g in range(model.ngeom):
        if int(model.geom_group[g]) not in groups:
            continue
        mesh = _geom_mesh(model, g)
        if mesh is None:
            continue
        pose = np.eye(4)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        pose[:3, :3], pose[:3, 3] = rot.reshape(3, 3), model.geom_pos[g]
        mesh.apply_transform(pose)
        colour = (np.clip(geom_rgba(model, g), 0, 1) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile(colour, (len(mesh.vertices), 1)))
        parts.setdefault(int(model.geom_bodyid[g]), []).append(mesh)
    return {b: (p[0] if len(p) == 1 else trimesh.util.concatenate(p)) for b, p in parts.items()}


def pack_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.concatenate([[data.time], data.xpos.ravel(), data.xquat.ravel()]).astype(np.float32)


def unpack_state(state) -> tuple[float, np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    n = (len(state) - 1) // 7
    return float(state[0]), state[1:1 + 3 * n].reshape(n, 3), state[1 + 3 * n:].reshape(n, 4)
