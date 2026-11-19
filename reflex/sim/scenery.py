"""Optional outdoor scenery for the simulator: a park, built from code.

The default scene ("lab": checkered floor, a wall, coloured boxes) stays as it is; this
is an extra, chosen with `scene="park"`. Nothing is downloaded: every texture is noise
generated here, and every object is a MuJoCo primitive.

What it is for: a less sterile picture, and a more honest event stream when the dog
moves (grass, bricks and bark all produce events under ego-motion, flat colours do not).
What it is not: a photo-realistic or measured environment. The network was trained on
the lab scene, so numbers obtained here are not comparable with the ones in the README.

Layout, in the dog frame (+x forward, +y left): the dog stands on a gravel path that runs
along x. The lane where balls and walkers move (x from 0 to 6 m, |y| under 3.5 m) is
kept free of obstacles.
"""
from __future__ import annotations

import mujoco
import numpy as np

SCENES = ("lab", "park")
TEX = 256      # texture side in pixels


def _noise(rng: np.random.Generator, size: int, scale: float) -> np.ndarray:
    """Tileable smooth noise in [0, 1]; `scale` is the blob size in pixels."""
    f = np.fft.fftfreq(size)
    r2 = f[:, None] ** 2 + f[None, :] ** 2
    spectrum = np.fft.fft2(rng.standard_normal((size, size))) * np.exp(-r2 * (scale * np.pi) ** 2)
    out = np.real(np.fft.ifft2(spectrum))
    out -= out.min()
    return out / max(out.max(), 1e-9)


def _rgb(base, variation: np.ndarray, amount: float) -> np.ndarray:
    """A base colour modulated by a [0, 1] map: uint8 image (H, W, 3)."""
    img = np.asarray(base, float)[None, None, :] * (1.0 + amount * (variation[..., None] - 0.5) * 2.0)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def textures(rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = TEX
    fine, mid, coarse = _noise(rng, n, 1.5), _noise(rng, n, 6), _noise(rng, n, 30)
    out = {
        "grass": _rgb([0.24, 0.42, 0.16], 0.5 * fine + 0.3 * mid + 0.2 * coarse, 0.75),
        "gravel": _rgb([0.62, 0.57, 0.48], 0.7 * fine + 0.3 * coarse, 0.45),
        "leaves": _rgb([0.16, 0.36, 0.13], 0.6 * mid + 0.4 * fine, 0.9),
    }
    # Bark and wood: noise stretched along one axis.
    streaks = np.repeat(_noise(rng, n, 2)[:, :1], n, axis=1) * 0.6 + 0.4 * fine
    out["bark"] = _rgb([0.33, 0.24, 0.16], streaks.T, 0.6)
    planks = (np.arange(n) // (n // 4) % 2)[:, None] * 0.25 + 0.75 * streaks
    out["wood"] = _rgb([0.50, 0.34, 0.20], planks, 0.4)
    # Bricks: rows offset by half a brick, light mortar lines.
    rows, cols = 8, 4
    y, x = np.mgrid[0:n, 0:n]
    row = y // (n // rows)
    xs = (x + (row % 2) * (n // cols // 2)) % (n // cols)
    mortar = (y % (n // rows) < 3) | (xs < 3)
    tone = np.random.default_rng(1).uniform(0.3, 0.9, (rows, cols * 2))[row, ((x + (row % 2) * (n // cols // 2)) // (n // cols)) % (cols * 2)]
    brick = _rgb([0.58, 0.27, 0.20], 0.6 * tone + 0.4 * fine, 0.4)
    brick[mortar] = (205, 200, 190)
    out["brick"] = brick
    # Facade: a plastered wall with a grid of dark windows.
    facade = _rgb([0.78, 0.73, 0.64], coarse, 0.12)
    window = ((y % (n // 4)) > n // 16) & ((y % (n // 4)) < n // 5) & ((x % (n // 4)) > n // 14) & ((x % (n // 4)) < n // 6)
    facade[window] = (38, 48, 62)
    out["facade"] = facade
    return out


def mean_colours(tex: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: (v.reshape(-1, 3).mean(0) / 255).tolist() for k, v in tex.items()}


def add_park(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    """Add ground, sky, light and objects of the park to `spec`. Uses its own random
    stream for the textures so the object layout depends only on `rng`."""
    G = mujoco.mjtGeom
    w = spec.worldbody
    tex = textures(np.random.default_rng(7))
    repeat = {"grass": [0.6, 0.6], "gravel": [0.8, 0.8], "leaves": [1.5, 1.5], "bark": [2, 1],
              "wood": [2, 2], "brick": [1.0, 1.0], "facade": [0.25, 0.25]}
    for name, img in tex.items():
        spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=TEX, height=TEX,
                         nchannel=3, data=img.ravel().tolist())
        mat = spec.add_material(name=name, texrepeat=repeat[name], texuniform=True)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.45, 0.65, 0.92], rgb2=[0.80, 0.87, 0.95], width=256, height=1536)

    # Sun: one directional light with shadows, low enough to draw long ones.
    w.add_light(name="sun", pos=[0, 0, 8], dir=[0.35, 0.45, -1.0], diffuse=[0.75, 0.72, 0.65],
                type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True)

    w.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[0, 0, 0.05], material="grass")
    # The path is paint: a few millimetres thick and without collisions.
    w.add_geom(name="path", type=G.mjGEOM_BOX, pos=[3.0, 0, 0.0], size=[9.0, 1.6, 0.003],
               material="gravel", contype=0, conaffinity=0)

    def tree(i: int, x: float, y: float) -> None:
        h, r = float(rng.uniform(1.8, 3.0)), float(rng.uniform(0.12, 0.22))
        w.add_geom(name=f"tree{i}_trunk", type=G.mjGEOM_CYLINDER, pos=[x, y, h / 2], size=[r, h / 2, 0],
                   material="bark")
        for k in range(3):
            off = rng.uniform(-0.5, 0.5, 2)
            s = rng.uniform(0.7, 1.3, 3)
            w.add_geom(name=f"tree{i}_crown{k}", type=G.mjGEOM_ELLIPSOID,
                       pos=[x + off[0], y + off[1], h + 0.4 + 0.5 * k], size=s.tolist(),
                       material="leaves", contype=0, conaffinity=0)

    # Trees along both sides of the path and a looser group at the far end.
    spots = [(x + rng.uniform(-0.6, 0.6), side * rng.uniform(4.2, 7.5))
             for side in (-1, 1) for x in (-3.0, 0.5, 4.0, 7.5, 11.0)]
    spots += [(rng.uniform(13, 17), rng.uniform(-7, 7)) for _ in range(4)]
    for i, (x, y) in enumerate(spots):
        tree(i, float(x), float(y))

    for i in range(8):      # bushes
        side = rng.choice([-1, 1])
        w.add_geom(name=f"bush{i}", type=G.mjGEOM_ELLIPSOID,
                   pos=[rng.uniform(-3, 12), side * rng.uniform(3.8, 6.5), 0.25],
                   size=[rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.5)],
                   material="leaves")

    def bench(i: int, x: float, y: float) -> None:
        face = -np.sign(y)      # the backrest is on the side away from the path
        w.add_geom(name=f"bench{i}_seat", type=G.mjGEOM_BOX, pos=[x, y, 0.42], size=[0.75, 0.22, 0.03], material="wood")
        w.add_geom(name=f"bench{i}_back", type=G.mjGEOM_BOX, pos=[x, y - face * 0.22, 0.72], size=[0.75, 0.03, 0.22],
                   material="wood")
        for k, dx in enumerate((-0.65, 0.65)):
            w.add_geom(name=f"bench{i}_leg{k}", type=G.mjGEOM_BOX, pos=[x + dx, y, 0.2], size=[0.04, 0.2, 0.2],
                       rgba=[0.15, 0.15, 0.16, 1])

    bench(0, 2.0, 3.9)
    bench(1, 5.5, -3.9)

    for i, (x, y) in enumerate(((-1.0, -2.6), (6.5, 2.6), (12.0, -2.6))):      # lamp posts
        w.add_geom(name=f"lamp{i}_post", type=G.mjGEOM_CYLINDER, pos=[x, y, 1.6], size=[0.05, 1.6, 0],
                   rgba=[0.12, 0.13, 0.14, 1])
        w.add_geom(name=f"lamp{i}_head", type=G.mjGEOM_SPHERE, pos=[x, y, 3.3], size=[0.16, 0, 0],
                   rgba=[0.95, 0.93, 0.80, 1], contype=0, conaffinity=0)

    # A low brick wall across the far end of the lane: it also stops the balls.
    w.add_geom(name="back_wall", type=G.mjGEOM_BOX, pos=[8.5, 0, 0.45], size=[0.12, 3.2, 0.45], material="brick")

    # Houses in the distance, so the horizon is not empty.
    for i, (x, y, half_w, h) in enumerate(((22, -9, 5, 4.0), (24, 3, 6, 5.5), (21, 13, 4, 3.5), (-14, 10, 5, 4.5))):
        w.add_geom(name=f"house{i}", type=G.mjGEOM_BOX, pos=[x, y, h], size=[3.0, half_w, h], material="facade")

    # A goal behind the dog (the keeper demo). Posts only, no collisions.
    for name, pos, size in (("post_l", [-0.7, 1.2, 0.5], [0.04, 0.5, 0]), ("post_r", [-0.7, -1.2, 0.5], [0.04, 0.5, 0])):
        w.add_geom(name=f"goal_{name}", type=G.mjGEOM_CAPSULE, pos=pos, size=size, rgba=[0.95, 0.95, 0.95, 1],
                   contype=0, conaffinity=0)
    w.add_geom(name="goal_bar", type=G.mjGEOM_CAPSULE, pos=[-0.7, 0, 1.0], size=[0.04, 1.2, 0],
               quat=[0.7071068, 0.7071068, 0, 0], rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)
