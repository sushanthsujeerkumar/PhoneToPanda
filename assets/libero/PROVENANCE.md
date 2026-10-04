# Bundled LIBERO scene assets

These files are a minimal unmodified subset of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, retrieved 3 October 2026:

- `libero/libero/assets/scenes/libero_tabletop_base_style.xml`
- Five tabletop/floor/wall PNGs from `libero/libero/assets/textures/`
- Upstream `LICENSE` (MIT), retained alongside the assets.

They provide the standard LIBERO table scene without downloading unrelated benchmark object meshes. `../cube.xml`, `../open_box.xml`, and `../pick_place.bddl` are PhoneToPanda's custom task assets. The Panda arm and gripper geometry comes from the installed robosuite package.
