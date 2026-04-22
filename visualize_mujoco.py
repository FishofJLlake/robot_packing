import time
import numpy as np
import mujoco
import mujoco.viewer


def get_palette_color(index):
    colors = [
        "0.8 0.25 0.25 1",  # Red
        "0.25 0.8 0.25 1",  # Green
        "0.25 0.25 0.8 1",  # Blue
        "0.8 0.8 0.25 1",   # Yellow
        "0.25 0.8 0.8 1",   # Cyan
        "0.8 0.25 0.8 1",   # Magenta
        "0.9 0.5 0.1 1",    # Orange
        "0.5 0.1 0.9 1",    # Purple
    ]
    return colors[index % len(colors)]


def _heightmap_to_collision_boxes_xml(
    initial_heightmap,
    cage_origin,
    resolution,
    max_boxes=2500,
    min_height=0.002,
):
    """
    Convert heightmap to static collision boxes with adaptive downsampling.

    This avoids creating too many tiny geoms when resolution is very high.
    """
    if initial_heightmap is None:
        return "", 0, 1

    hm = np.asarray(initial_heightmap, dtype=float)
    if hm.ndim != 2 or hm.size == 0:
        return "", 0, 1

    valid_count = int(np.count_nonzero(hm > min_height))
    if valid_count == 0:
        return "", 0, 1

    if max_boxes <= 0:
        stride = 1
    else:
        stride = max(1, int(np.ceil(np.sqrt(valid_count / float(max_boxes)))))

    rows, cols = hm.shape
    boxes_xml = []
    box_count = 0

    for r0 in range(0, rows, stride):
        r1 = min(rows, r0 + stride)
        for c0 in range(0, cols, stride):
            c1 = min(cols, c0 + stride)
            block = hm[r0:r1, c0:c1]
            h = float(np.nanmax(block))
            if not np.isfinite(h) or h <= min_height:
                continue

            bw = (c1 - c0) * resolution
            bl = (r1 - r0) * resolution
            bx = cage_origin[0] + (c0 + (c1 - c0) / 2.0) * resolution
            by = cage_origin[1] + (r0 + (r1 - r0) / 2.0) * resolution
            bz = cage_origin[2] + h / 2.0
            hz = h / 2.0

            boxes_xml.append(
                f'<geom type="box" pos="{bx} {by} {bz}" '
                f'size="{bw/2.0} {bl/2.0} {hz}" '
                f'rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>'
            )
            box_count += 1

    return "\n".join(boxes_xml), box_count, stride


def replay_packing_process(
    placed_items,
    cage_origin,
    cage_dims,
    initial_heightmap=None,
    resolution=0.01,
    max_environment_boxes=2500,
):
    """
    Use MuJoCo viewer to replay the full packing sequence.

    Parameters
    ----------
    placed_items : list
        Output from PackingPlanner (already placed items).
    cage_origin : [x, y, z]
    cage_dims : [w, l, h]
    initial_heightmap : np.ndarray or None
        Initial scene represented by static collision boxes.
    resolution : float
        Heightmap cell size in meters.
    max_environment_boxes : int
        Upper bound for generated static geoms from heightmap.
    """
    cw, cl, ch = cage_dims
    cx = cage_origin[0] + cw / 2
    cy = cage_origin[1] + cl / 2
    cz = cage_origin[2] + ch / 2

    # Build environment collision geoms from initial heightmap (adaptive coarsening).
    env_str, env_box_count, env_stride = _heightmap_to_collision_boxes_xml(
        initial_heightmap=initial_heightmap,
        cage_origin=cage_origin,
        resolution=resolution,
        max_boxes=max_environment_boxes,
    )

    # Dynamic item bodies.
    bodies_xml = []
    for i, item in enumerate(placed_items):
        L, W, H = item["dimensions"]
        hx, hy, hz = L / 2.0, W / 2.0, H / 2.0

        hold_x = 50 + (i % 5) * 1.5
        hold_y = (i // 5) * 1.5
        hold_z = -4.5

        c_str = get_palette_color(i)

        body = f"""
        <body name="item_{i}" pos="{hold_x} {hold_y} {hold_z}">
            <joint type="free" damping="20.0" armature="0.01"/>
            <geom type="box" size="{hx} {hy} {hz}" mass="50.0" rgba="{c_str}"
                  friction="1.0 0.05 0.001" contype="1" conaffinity="1"/>
        </body>
        """
        bodies_xml.append(body)

    dynamic_bodies = "\n".join(bodies_xml)

    xml = f"""
    <mujoco>
        <size memory="64M"/>
        <option timestep="0.002" gravity="0 0 -9.81"/>
        <visual>
            <global azimuth="140" elevation="-30"/>
        </visual>
        <worldbody>
            <light pos="{cx} {cy} {ch+2}" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>

            <geom type="plane" size="5 5 0.1" pos="{cx} {cy} 0" rgba="0.9 0.9 0.9 1" contype="1" conaffinity="1"/>

            <body name="cage_walls" pos="{cx} {cy} {cz}">
                <geom type="box" size="0.02 {cl/2} {ch/2}" pos="{-cw/2} 0 0" rgba="0.2 0.5 0.8 0.15" contype="1" conaffinity="1"/>
                <geom type="box" size="0.02 {cl/2} {ch/2}" pos="{cw/2} 0 0" rgba="0.2 0.5 0.8 0.15" contype="1" conaffinity="1"/>
                <geom type="box" size="{cw/2} 0.02 {ch/2}" pos="0 {cl/2} 0" rgba="0.2 0.5 0.8 0.15" contype="1" conaffinity="1"/>
                <geom type="box" size="{cw/2} 0.02 {ch/10}" pos="0 {-cl/2} {-ch/2 + ch/10}" rgba="0.8 0.2 0.2 0.2" contype="1" conaffinity="1"/>
            </body>

            {env_str}

            <geom type="plane" size="20 20 0.1" pos="50 0 -5" rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1"/>

            {dynamic_bodies}
        </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    model.stat.extent = max(cw, cl, ch) * 2.5
    model.stat.center[:] = [cx, cy, ch / 3.0]

    print("==========================================")
    print(" 启动 MuJoCo 查看器... (按 ESC 退出)")
    if env_box_count > 0:
        print(
            f" 初始环境碰撞盒: {env_box_count} "
            f"(heightmap stride={env_stride}, max={max_environment_boxes})"
        )
    print("==========================================")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(len(placed_items)):
            if not viewer.is_running():
                break

            item = placed_items[i]
            pose = item["result"]["pose"]

            px, py, pz = pose["position"]
            drop_pz = pz + 0.005

            qx, qy, qz, qw = pose["quaternion"]  # scipy format

            body = model.body(f"item_{i}")
            jnt_id = body.jntadr[0]
            qpos_idx = model.jnt_qposadr[jnt_id]
            qvel_idx = model.jnt_dofadr[jnt_id]

            data.qpos[qpos_idx:qpos_idx + 3] = [px, py, drop_pz]
            data.qpos[qpos_idx + 3:qpos_idx + 7] = [qw, qx, qy, qz]
            data.qvel[qvel_idx:qvel_idx + 6] = 0

            print(f"[{i+1}/{len(placed_items)}] 正在投放物品 (w={item['dimensions']})")

            steps_to_wait = int(1.5 / 0.002)
            for _ in range(steps_to_wait):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)
                if not viewer.is_running():
                    break

            for _ in range(int(0.5 / 0.002)):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)

        print("所有物品投放完毕！您可以旋转视角查看堆垛结果。")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.016)


if __name__ == "__main__":
    from test_packing import test_continuous_packing
    from config import CAGE_WIDTH, CAGE_LENGTH, CAGE_HEIGHT

    print("准备使用 test_packing 的连续装箱测试数据驱动物理场景...")

    import test_packing

    planner = test_packing.test_continuous_packing()

    if planner and hasattr(planner, "placed_items"):
        replay_packing_process(
            planner.placed_items,
            [0, 0, 0],
            [CAGE_WIDTH, CAGE_LENGTH, CAGE_HEIGHT],
        )
    else:
        print("没找到任何被放置的货物，回放退出。")
