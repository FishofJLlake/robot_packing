import numpy as np
import mujoco

class MujocoSimulator:
    def __init__(self, cage_origin, cage_width, cage_length, cage_height, resolution):
        self.cage_origin = np.array(cage_origin)
        self.w = cage_width
        self.l = cage_length
        self.h = cage_height
        self.res = resolution

    def simulate_tilt(self, full_heightmap, item_dims, initial_pos, initial_quat):
        """
        Runs a physics simulation to find the rest pose of a tilted object.
        Returns:
            is_inside_cage: bool
            final_pos: numpy array (3,)
            final_quat: numpy array (4,) [x, y, z, w]
        """
        grid_rows, grid_cols = full_heightmap.shape
        
        # Calculate search window bounds (approx. +/- 0.4 meters from center)
        cx_grid = int((initial_pos[0] - self.cage_origin[0]) / self.res)
        cy_grid = int((initial_pos[1] - self.cage_origin[1]) / self.res)
        
        pad = int(0.4 / self.res)
        min_r = max(0, cy_grid - pad)
        max_r = min(grid_rows, cy_grid + pad)
        min_c = max(0, cx_grid - pad)
        max_c = min(grid_cols, cx_grid + pad)
        
        boxes_xml = []
        for r in range(min_r, max_r):
            for c in range(min_c, max_c):
                h = full_heightmap[r, c]
                if h > 0.001:
                    cx = self.cage_origin[0] + (c + 0.5) * self.res
                    cy = self.cage_origin[1] + (r + 0.5) * self.res
                    cz = self.cage_origin[2] + h / 2.0
                    hx = self.res / 2.0
                    hy = self.res / 2.0
                    hz = h / 2.0
                    boxes_xml.append(f'<geom type="box" pos="{cx} {cy} {cz}" size="{hx} {hy} {hz}" rgba="0.5 0.5 0.5 1" contype="1" conaffinity="1"/>')
        
        boxes_str = "\n".join(boxes_xml)
        
        item_hx, item_hy, item_hz = np.array(item_dims) / 2.0
        
        # scipy.spatial.transform is x, y, z, w. mujoco is w, x, y, z
        q_x, q_y, q_z, q_w = initial_quat
        mj_quat = f"{q_w} {q_x} {q_y} {q_z}"
        
        px, py, pz = initial_pos
        
        xml = f"""
        <mujoco>
            <option timestep="0.005" gravity="0 0 -9.81"/>
            <worldbody>
                <light pos="0 0 3" dir="0 0 -1" directional="true"/>
                <!-- Floor -->
                <geom type="plane" size="{self.w} {self.l} 0.1" pos="{self.cage_origin[0]+self.w/2} {self.cage_origin[1]+self.l/2} {self.cage_origin[2]}" rgba="0.8 0.9 0.8 1" contype="1" conaffinity="1"/>
                
                <!-- Heightmap boxes -->
                {boxes_str}
                
                <!-- The falling item -->
                <body name="item" pos="{px} {py} {pz}" quat="{mj_quat}">
                    <freejoint/>
                    <geom type="box" size="{item_hx} {item_hy} {item_hz}" mass="2.0" rgba="1 0 0 1" contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        try:
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
        except Exception as e:
            print(f"Mujoco XML string error: {e}")
            return False, initial_pos, initial_quat
        
        max_steps = 1000  # 5 seconds at 0.005
        tolerance_vel = 1e-3
        
        for step in range(max_steps):
            mujoco.mj_step(model, data)
            vel = np.linalg.norm(data.qvel[:3])
            omega = np.linalg.norm(data.qvel[3:])
            if step > 100 and vel < tolerance_vel and omega < tolerance_vel:
                break
                
        final_pos = np.array(data.qpos[:3])
        fw, fx, fy, fz = data.qpos[3:7]
        final_quat = np.array([fx, fy, fz, fw])
        
        is_inside = True
        if final_pos[2] < self.cage_origin[2] + item_hz * 0.1: # Very low
            is_inside = False
            
        min_x = self.cage_origin[0] - 0.1
        max_x = self.cage_origin[0] + self.w + 0.1
        min_y = self.cage_origin[1] - 0.1
        max_y = self.cage_origin[1] + self.l + 0.1
        
        if not (min_x <= final_pos[0] <= max_x):
            is_inside = False
        if not (min_y <= final_pos[1] <= max_y):
            is_inside = False
            
        return is_inside, final_pos, final_quat
