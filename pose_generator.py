"""
3D装箱系统 — 6D位姿生成

根据高度图坐标、朝向信息、底面倾斜微调，生成最终的6D位姿。
"""
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation


def compute_6d_pose(
    cage_origin: np.ndarray,
    row: int, col: int,
    item_grid_rows: int, item_grid_cols: int,
    place_height: float,
    item_up_dim: float,
    orientation: dict,
    tilt_roll: float = 0.0,
    tilt_pitch: float = 0.0,
    resolution: float = 0.01,
) -> dict:
    """
    根据候选位置信息计算最终6D位姿。
    
    Parameters
    ----------
    cage_origin : np.ndarray, shape (3,)
        笼体原点坐标 (x_min, y_min, z_min)。
    row, col : int
        高度图上的起始行列（货物左下角对应的格子）。
    item_grid_rows, item_grid_cols : int
        货物在高度图上占的行数和列数。
    place_height : float
        放置高度（底面到笼底的高度，米）。
    item_up_dim : float
        该朝向下货物的高度（向上方向的尺寸，米）。
    orientation : dict
        朝向信息, 包含 'roll', 'pitch', 'yaw'。
    tilt_roll, tilt_pitch : float
        底面倾斜微调角度（弧度）。
    resolution : float
        高度图分辨率（米/格）。
    
    Returns
    -------
    dict:
        'position': (x, y, z)       物体中心坐标
        'orientation': (roll, pitch, yaw)   欧拉角（弧度）
        'rotation_matrix': np.ndarray (3,3)  旋转矩阵
        'quaternion': np.ndarray (4,)        四元数 (x, y, z, w)
    """
    # 计算物体中心在世界坐标中的位置
    x_center = cage_origin[0] + (col + item_grid_cols / 2.0) * resolution
    y_center = cage_origin[1] + (row + item_grid_rows / 2.0) * resolution
    z_center = cage_origin[2] + place_height + item_up_dim / 2.0
    
    # 合成最终旋转角
    base_roll  = orientation['roll']
    base_pitch = orientation['pitch']
    base_yaw   = orientation['yaw']
    
    final_roll  = base_roll  + tilt_roll
    final_pitch = base_pitch + tilt_pitch
    final_yaw   = base_yaw
    
    # 使用 scipy Rotation 计算旋转矩阵和四元数
    # 旋转顺序: 先Yaw(Z) → Pitch(Y) → Roll(X)，使用外旋（extrinsic）
    # scipy的 'xyz' 是内旋（intrinsic），等价于 'ZYX' 外旋
    # 我们用 'XYZ' 外旋 = 'xyz' 内旋的逆序，即 from_euler('ZYX', [yaw, pitch, roll])
    rot = Rotation.from_euler('ZYX', [final_yaw, final_pitch, final_roll])
    rot_matrix = rot.as_matrix()
    quat = rot.as_quat()  # (x, y, z, w)
    
    return {
        'position': (x_center, y_center, z_center),
        'orientation_euler': (final_roll, final_pitch, final_yaw),
        'rotation_matrix': rot_matrix,
        'quaternion': quat,
    }


def pose_to_transform_matrix(pose: dict) -> np.ndarray:
    """
    将位姿转换为4x4齐次变换矩阵。
    
    Parameters
    ----------
    pose : dict
        compute_6d_pose 的返回值。
    
    Returns
    -------
    np.ndarray, shape (4, 4)
    """
    T = np.eye(4)
    T[:3, :3] = pose['rotation_matrix']
    T[0, 3] = pose['position'][0]
    T[1, 3] = pose['position'][1]
    T[2, 3] = pose['position'][2]
    return T


def format_pose_string(pose: dict) -> str:
    """格式化输出位姿信息。"""
    pos = pose['position']
    euler = pose['orientation_euler']
    return (
        f"位置: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m\n"
        f"欧拉角 (Roll, Pitch, Yaw): "
        f"({np.rad2deg(euler[0]):.2f}°, {np.rad2deg(euler[1]):.2f}°, {np.rad2deg(euler[2]):.2f}°)\n"
        f"四元数 (x,y,z,w): ({pose['quaternion'][0]:.6f}, {pose['quaternion'][1]:.6f}, "
        f"{pose['quaternion'][2]:.6f}, {pose['quaternion'][3]:.6f})"
    )
