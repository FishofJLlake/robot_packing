"""
3D装箱系统 — 3D可视化模块

使用 Open3D 和 Matplotlib 进行装箱结果可视化。
"""
import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# 预定义的颜色表（为不同货物使用不同颜色）
COLORS = [
    [0.85, 0.33, 0.33],  # 红
    [0.33, 0.65, 0.85],  # 蓝
    [0.33, 0.85, 0.45],  # 绿
    [0.85, 0.75, 0.33],  # 黄
    [0.65, 0.33, 0.85],  # 紫
    [0.85, 0.55, 0.33],  # 橙
    [0.33, 0.85, 0.85],  # 青
    [0.85, 0.33, 0.65],  # 粉
]


def visualize_heightmap(heightmap: np.ndarray, 
                        title: str = "高度图",
                        save_path: Optional[str] = None):
    """
    用 matplotlib 热力图显示高度图。
    
    Parameters
    ----------
    heightmap : np.ndarray
        高度图矩阵。
    title : str
        图表标题。
    save_path : str, optional
        保存路径。
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(
        heightmap, 
        cmap='YlOrRd', 
        origin='lower',
        aspect='equal',
        interpolation='nearest'
    )
    
    ax.set_xlabel('X (宽度方向, 左→右)')
    ax.set_ylabel('Y (深度方向, 外→里)')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('高度 (m)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"高度图已保存: {save_path}")
    
    plt.show()


def visualize_packing_2d(heightmap: np.ndarray, 
                          placed_items: List[dict],
                          title: str = "装箱俯视图",
                          save_path: Optional[str] = None):
    """
    俯视图显示已放置的货物轮廓叠加在高度图上。
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(
        heightmap, 
        cmap='Greys', 
        origin='lower',
        aspect='equal',
        alpha=0.6,
        interpolation='nearest'
    )
    
    for i, item in enumerate(placed_items):
        result = item['result']
        row, col = result['grid_pos']
        rows, cols = result['item_grid_size']
        
        color = COLORS[i % len(COLORS)]
        
        # 绘制矩形轮廓
        rect = plt.Rectangle(
            (col, row), cols, rows,
            linewidth=2,
            edgecolor=color,
            facecolor=color + [0.3],  # 半透明填充
        )
        ax.add_patch(rect)
        
        # 标注编号
        ax.text(
            col + cols / 2, row + rows / 2, 
            f'#{i+1}',
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color='black'
        )
    
    ax.set_xlabel('X (宽度方向, 左→右)')
    ax.set_ylabel('Y (深度方向, 外→里)')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('高度 (m)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"俯视图已保存: {save_path}")
    
    plt.show()


def create_cage_wireframe(origin, width, length, height, color=[1.0, 0.5, 0.0]):
    """
    创建笼车线框（Open3D LineSet）。
    """
    if not HAS_OPEN3D:
        return None
    
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + width, y0 + length, z0 + height
    
    points = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # 底面
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # 顶面
    ]
    
    lines = [
        [0,1], [1,2], [2,3], [3,0],  # 底
        [4,5], [5,6], [6,7], [7,4],  # 顶
        [0,4], [1,5], [2,6], [3,7],  # 竖边
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines  = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return line_set


def create_box_mesh(center, half_extents, rotation_matrix, color):
    """
    创建一个旋转的长方体网格（Open3D TriangleMesh）。
    
    Parameters
    ----------
    center : (3,) 中心坐标
    half_extents : (3,) 半边长 (dx, dy, dz)
    rotation_matrix : (3,3) 旋转矩阵
    color : (3,) RGB颜色 [0,1]
    """
    if not HAS_OPEN3D:
        return None
    
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=half_extents[0] * 2,
        height=half_extents[1] * 2,
        depth=half_extents[2] * 2,
    )
    
    # create_box是从(0,0,0)到(w,h,d)，需要平移到中心
    mesh.translate(-np.array(half_extents))
    mesh.rotate(rotation_matrix, center=np.zeros(3))
    mesh.translate(center)
    
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    
    return mesh


def visualize_packing_3d(cage_origin: Tuple[float, float, float],
                          cage_width: float,
                          cage_length: float,
                          cage_height: float,
                          placed_items: List[dict],
                          point_cloud: Optional[np.ndarray] = None,
                          point_colors: Optional[np.ndarray] = None,
                          title: str = "3D装箱结果"):
    """
    使用Open3D的3D可视化窗口展示装箱结果。
    
    Parameters
    ----------
    cage_origin : (3,) 笼体原点
    cage_width, cage_length, cage_height : float 笼体尺寸
    placed_items : list of dict
        每个元素包含 'dimensions' 和 'result' (plan_placement的返回值)
    point_cloud : np.ndarray, optional
        原始点云数据（用于背景显示）
    point_colors : np.ndarray, optional
        原始点云颜色数据
    """
    if not HAS_OPEN3D:
        print("Open3D未安装，跳过3D可视化。使用 pip install open3d 安装。")
        return
    
    geometries = []
    
    # 1. 笼车线框
    cage = create_cage_wireframe(
        cage_origin, cage_width, cage_length, cage_height
    )
    if cage:
        geometries.append(cage)
    
    # 2. 原始点云（如果有）
    if point_cloud is not None and len(point_cloud) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        if point_colors is not None and len(point_colors) == len(point_cloud):
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
        else:
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd)
    
    # 3. 已放置的货物
    for i, item in enumerate(placed_items):
        result = item['result']
        pose = result['pose']
        ori = result['orientation']
        
        base_x, base_y = ori['base_dims']
        up_dim = ori['up_dim']
        
        half_extents = np.array([base_x / 2, base_y / 2, up_dim / 2])
        center = np.array(pose['position'])
        rot_matrix = pose['rotation_matrix']
        
        color = COLORS[i % len(COLORS)]
        
        # 创建货物网格，使用AABB因为旋转已经在pose中了
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=base_x, height=base_y, depth=up_dim
        )
        # 移到原点居中
        mesh.translate([-base_x/2, -base_y/2, -up_dim/2])
        # 应用旋转
        sim_pose = result.get('simulated_pose')
        if sim_pose:
            # 基础尺寸已对应朝向，但物理引擎的位姿使用的是物品原尺寸，所以我们需要重新定义Box。
            # 或者，因为已经按照base_x, base_y生成，我们需要通过旋转恢复物理模拟引擎给它的真实转角。
            # 为简单起见，如果使用仿真位姿，直接运用仿真的 rotation matrix和位置
            
            # 使用真实的原始物品尺寸
            orig_dim_L, orig_dim_W, orig_dim_H = item['dimensions']
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=orig_dim_L, height=orig_dim_W, depth=orig_dim_H
            )
            mesh.translate([-orig_dim_L/2, -orig_dim_W/2, -orig_dim_H/2])
            mesh.rotate(sim_pose['rotation_matrix'], center=np.zeros(3))
            center = sim_pose['position']
        else:
            # 没有仿真，或者没有倾覆
            # 基础倾斜微调
            # mesh.rotate(pose['rotation_matrix'], center=np.zeros(3))
            pass
        
        # 平移到放置位置
        mesh.translate(center)
        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        geometries.append(mesh)
        
        # 添加编号文字（坐标轴标记）
        # Open3D不直接支持文字，用小球标记
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(center + np.array([0, 0, up_dim/2 + 0.02]))
        sphere.paint_uniform_color(color)
        geometries.append(sphere)
    
    # 4. 坐标轴
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(coord)
    
    # 显示
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"  已放置 {len(placed_items)} 个货物")
    print(f"{'='*50}")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=720,
    )
