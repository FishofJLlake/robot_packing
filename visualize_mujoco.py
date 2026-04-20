import time
import numpy as np
import mujoco
import mujoco.viewer
import json

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

def replay_packing_process(placed_items, cage_origin, cage_dims, initial_heightmap=None, resolution=0.01):
    """
    使用 MuJoCo viewer 回放整个装箱序列。
    placed_items: 由 PackingPlanner 导出的已放置货物列表。
    cage_origin: [x, y, z]
    cage_dims: [w, l, h]
    """
    cw, cl, ch = cage_dims
    cx = cage_origin[0] + cw/2
    cy = cage_origin[1] + cl/2
    cz = cage_origin[2] + ch/2
    
    # 0. 将初始高度图环境注入为静态碰撞箱
    environment_boxes_xml = []
    if initial_heightmap is not None:
        grid_rows, grid_cols = initial_heightmap.shape
        hx = resolution / 2.0
        hy = resolution / 2.0
        for r in range(grid_rows):
            for c in range(grid_cols):
                h = initial_heightmap[r, c]
                if h > 0.001:
                    bx = cage_origin[0] + (c + 0.5) * resolution
                    by = cage_origin[1] + (r + 0.5) * resolution
                    bz = cage_origin[2] + h / 2.0
                    hz = h / 2.0
                    environment_boxes_xml.append(
                        f'<geom type="box" pos="{bx} {by} {bz}" size="{hx} {hy} {hz}" '
                        f'rgba="0.6 0.6 0.6 1" contype="1" conaffinity="1"/>'
                    )
    env_str = "\n".join(environment_boxes_xml)
    
    # 1. 动态生成 XML
    bodies_xml = []
    for i, item in enumerate(placed_items):
        L, W, H = item['dimensions']
        hx, hy, hz = L/2.0, W/2.0, H/2.0
        
        # 安排在遥远的准备区
        hold_x = 50 + (i % 5)*1.5
        hold_y = (i // 5)*1.5
        hold_z = -4.5
        
        c_str = get_palette_color(i)
        
        # 采用高重量，以及给自由关节加极大的阻尼避免弹飞乱飞
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
        <option timestep="0.002" gravity="0 0 -9.81"/>
        <visual>
            <global azimuth="140" elevation="-30"/>
        </visual>
        <worldbody>
            <light pos="{cx} {cy} {ch+2}" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
            
            <!-- 地面 -->
            <geom type="plane" size="5 5 0.1" pos="{cx} {cy} 0" rgba="0.9 0.9 0.9 1" contype="1" conaffinity="1"/>
            
            <!-- 笼车边框 (半透明，仅侧边) -->
            <body name="cage_walls" pos="{cx} {cy} {cz}">
                <!-- 左墙 -->
                <geom type="box" size="0.02 {cl/2} {ch/2}" pos="{-cw/2} 0 0" rgba="0.2 0.5 0.8 0.15" contype="1" conaffinity="1"/>
                <!-- 右墙 -->
                <geom type="box" size="0.02 {cl/2} {ch/2}" pos="{cw/2} 0 0" rgba="0.2 0.5 0.8 0.15" contype="1" conaffinity="1"/>
                <!-- 后墙 -->
                <geom type="box" size="{cw/2} 0.02 {ch/2}" pos="0 {cl/2} 0" rgba="0.2 0.5 0.8 0.15" contype="1" conaffinity="1"/>
                <!-- 前墙底挡板（防止向外滑出） -->
                <geom type="box" size="{cw/2} 0.02 {ch/10}" pos="0 {-cl/2} {-ch/2 + ch/10}" rgba="0.8 0.2 0.2 0.2" contype="1" conaffinity="1"/>
            </body>
            
            <!-- 初始环境障碍物 (来自于 PLY 点云) -->
            {env_str}
            
            <!-- 待命区地板 -->
            <geom type="plane" size="20 20 0.1" pos="50 0 -5" rgba="0.3 0.3 0.3 1" contype="1" conaffinity="1"/>
            
            {dynamic_bodies}
        </worldbody>
    </mujoco>
    """
    
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # 把视野移向笼子
    model.stat.extent = max(cw, cl, ch) * 2.5
    model.stat.center[:] = [cx, cy, ch/3.0]
    
    print("==========================================")
    print(" 启动 MuJoCo 查看器... （按 ESC 退出）")
    print("==========================================")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # 播放流程控制
        for i in range(len(placed_items)):
            if not viewer.is_running():
                break
                
            item = placed_items[i]
            pose = item['result']['pose']
            
            # 使用初始给出的期望投放位姿进行下落
            # 稍稍抬高一点点，贴近表面释放且不施加额外冲击
            px, py, pz = pose['position']
            drop_pz = pz + 0.005 # 仅抬高1公分
            
            # 兼容：如果有 sim_pose 甚至可以从那一侧降落
            # 但我们要展示物理动态过程，还是从标准 pose 落下
            
            qx, qy, qz, qw = pose['quaternion'] # scipy format
            
            # 获取对应的物理实例
            body_id = model.body(f"item_{i}").id
            jnt_id = model.body(f"item_{i}").jntadr[0]
            qpos_idx = model.jnt_qposadr[jnt_id]
            qvel_idx = model.jnt_dofadr[jnt_id]
            
            data.qpos[qpos_idx : qpos_idx+3] = [px, py, drop_pz]
            data.qpos[qpos_idx+3 : qpos_idx+7] = [qw, qx, qy, qz] # mujoco expects [w, x, y, z]
            data.qvel[qvel_idx : qvel_idx+6] = 0
            
            print(f"[{i+1}/{len(placed_items)}] 正在投放物品 (w={item['dimensions']})")
            
            # 让物理引擎运行一段时间（约 1.5 秒物理时间）
            steps_to_wait = int(1.5 / 0.002)
            for _ in range(steps_to_wait):
                mujoco.mj_step(model, data)
                viewer.sync()
                # 与真实时间同步（可在此加速减速，此处0.002为原速）
                time.sleep(0.002)
                if not viewer.is_running():
                    break
                    
            # 结束后允许再停歇缓冲0.5秒
            for _ in range(int(0.5 / 0.002)):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)
                
        # 所有物品投放完毕，挂起等待用户欣赏，直到关闭窗口
        print("所有物品投放完毕！您可以随意转动视角欣赏堆垛结果。")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.016)

# 若直接执行此文件可读取测试用例
if __name__ == "__main__":
    from test_packing import test_continuous_packing
    from config import CAGE_WIDTH, CAGE_LENGTH, CAGE_HEIGHT
    
    print("准备使用 test_packing 的连续装箱测试数据驱动物理场景...")
    
    # 给 test_packing 一点 patch，让它带回放置数组
    import test_packing
    planner = test_packing.test_continuous_packing()
    
    if planner and hasattr(planner, 'placed_items'):
        replay_packing_process(planner.placed_items, [0,0,0], [CAGE_WIDTH, CAGE_LENGTH, CAGE_HEIGHT])
    else:
        print("没找到任何被放置的货物，回放退出。")
