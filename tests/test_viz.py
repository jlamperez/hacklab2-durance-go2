

# --- web viewer helpers --------------------------------------------------------------------
def test_web_scene_meshes_and_state_roundtrip():
    import mujoco
    import numpy as np

    from reflex.sim.scene import Go2Sim
    from reflex.viz.web_scene import body_meshes, pack_state, unpack_state
    sim = Go2Sim(seed=0)
    meshes = body_meshes(sim.model)
    ball, base = sim.model.body("ball").id, sim.model.body("base").id
    assert ball in meshes and base in meshes and 0 in meshes          # world body: wall and clutter
    assert abs(meshes[ball].extents.max() - 0.2) < 0.01               # the ball, 10 cm radius
    assert tuple(meshes[ball].visual.vertex_colors[0][:3]) == (255, 114, 25)   # its own orange, not the material's white
    # Collision shapes (group 3) are left out: the trunk is the visual mesh, not boxes.
    assert len(meshes[base].vertices) > 1000
    t, pos, quat = unpack_state(pack_state(sim.model, sim.data))
    assert abs(t - sim.time) < 1e-4
    assert np.allclose(pos, sim.data.xpos, atol=1e-5) and np.allclose(quat, sim.data.xquat, atol=1e-5)
    assert pos.shape == (sim.model.nbody, 3)
    mujoco.mj_forward(sim.model, sim.data)


def test_web_scene_park_scenery_is_textured():
    import numpy as np

    from reflex.sim.scene import build_model
    from reflex.viz.web_scene import body_meshes, world_meshes
    model = build_model(np.random.default_rng(0), "park")
    scenery = world_meshes(model)
    assert {"grass", "gravel", "bark", "leaves", "brick"} <= set(scenery)
    grass = scenery["grass"]
    assert grass.visual.uv.shape == (len(grass.vertices), 2)
    assert grass.extents[0] > 20                       # the ground plane became a large square
    assert len(scenery["leaves"].faces) > 1000         # tree crowns (ellipsoids) are drawn
    assert len(body_meshes(model)[0].faces) > 1000
