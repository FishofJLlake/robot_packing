# 点云预处理与高度图生成优化

## 变更概览

对 `point_cloud_processor.py` 和相关文件进行了 4 项改进：

### 1. Voxel Downsample（体素下采样）

```diff:config.py
"""
3D装箱系统 — 全局配置与超参数
所有参数均可根据实际场景调整。
"""
import numpy as np

# ============================================================
# 货笼物理尺寸（米） — 超参数
# ============================================================
CAGE_LENGTH = 1.0       # Y方向（里-外，depth）
CAGE_WIDTH  = 0.8       # X方向（左-右，width）
CAGE_HEIGHT = 1.2       # Z方向（高度）

# ============================================================
# 高度图分辨率（米/格）
# ============================================================
HEIGHTMAP_RESOLUTION = 0.01   # 1 cm per cell

# ============================================================
# 货物间隙（米） — 超参数
# ============================================================
PLACEMENT_GAP = 0.01          # 1 cm 间隙，方便实际放置

# ============================================================
# 稳定性阈值
# ============================================================
MIN_SUPPORT_RATIO = 0.60      # 底部至少 60% 面积有支撑
MAX_TILT_ANGLE    = 5.0       # 最大允许倾斜角度（度）
SUPPORT_HEIGHT_TOLERANCE = 0.005  # 支撑高度容差（5mm以内视为有支撑）

# ============================================================
# 评分权重（体现先里后外、先左后右、先下后上）
# ============================================================
WEIGHT_DEPTH = 10.0    # Y方向（越里越好，越大分越高）
WEIGHT_LEFT  = 5.0     # X方向（越左越好）
WEIGHT_LOW   = 8.0     # Z方向（越低越好）
WEIGHT_FIT   = 3.0     # 紧凑度奖励（与已有货物紧贴）

# ============================================================
# 点云预处理参数
# ============================================================
STATISTICAL_OUTLIER_NB_NEIGHBORS = 20   # 统计去噪邻居数
STATISTICAL_OUTLIER_STD_RATIO   = 2.0   # 标准差倍数

# ============================================================
# 物体坐标系约定
# ============================================================
# 物体在传送带上的初始姿态（自然放置）:
#   物体坐标系原点 = 几何中心
#   物体X轴 = 长度(L)方向
#   物体Y轴 = 宽度(W)方向
#   物体Z轴 = 高度(H)方向（朝上）
#   初始位姿: Roll=0, Pitch=0, Yaw=0

# 6种放置朝向表
# 格式: (底面尺寸在世界X方向, 底面尺寸在世界Y方向, 放置高度, roll, pitch, yaw)
# 世界坐标系: X=笼宽方向, Y=笼深方向, Z=向上
ORIENTATIONS = [
    # ID  底面X    底面Y     up    (roll,       pitch,       yaw)         描述
    #  0  L        W         H     (0,          0,           0)           自然放置
    #  1  W        L         H     (0,          0,           pi/2)        绕Z轴90°
    #  2  L        H         W     (pi/2,       0,           0)           绕X轴90°
    #  3  H        L         W     (pi/2,       0,           pi/2)        绕X轴90° + 绕Z轴90°
    #  4  W        H         L     (0,          -pi/2,       0)           绕Y轴-90° (等价绕Y轴90°使Z朝向-X再调整)
    #  5  H        W         L     (0,          -pi/2,       pi/2)        绕Y轴-90° + 绕Z轴90°
]

def get_orientations(L, W, H):
    """
    给定物体原始尺寸 (L, W, H)，返回所有6种朝向下的参数。
    
    Returns
    -------
    list of dict, 每个dict包含:
        'base_dims': (base_x, base_y)  底面在世界坐标系中的尺寸
        'up_dim': float                放置后的高度
        'roll': float                  弧度
        'pitch': float                 弧度
        'yaw': float                   弧度
        'desc': str                    描述
    """
    orientations = [
        {
            'base_dims': (L, W), 'up_dim': H,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'desc': '自然放置 (LxW底, H高)'
        },
        {
            'base_dims': (W, L), 'up_dim': H,
            'roll': 0.0, 'pitch': 0.0, 'yaw': np.pi / 2,
            'desc': '绕Z轴90° (WxL底, H高)'
        },
        {
            'base_dims': (L, H), 'up_dim': W,
            'roll': np.pi / 2, 'pitch': 0.0, 'yaw': 0.0,
            'desc': '绕X轴90° (LxH底, W高)'
        },
        {
            'base_dims': (H, L), 'up_dim': W,
            'roll': np.pi / 2, 'pitch': 0.0, 'yaw': np.pi / 2,
            'desc': '绕X轴90°+绕Z轴90° (HxL底, W高)'
        },
        {
            'base_dims': (W, H), 'up_dim': L,
            'roll': 0.0, 'pitch': -np.pi / 2, 'yaw': 0.0,
            'desc': '绕Y轴-90° (WxH底, L高)'
        },
        {
            'base_dims': (H, W), 'up_dim': L,
            'roll': 0.0, 'pitch': -np.pi / 2, 'yaw': np.pi / 2,
            'desc': '绕Y轴-90°+绕Z轴90° (HxW底, L高)'
        },
    ]
    
    # 去重：如果L==W或W==H等，某些朝向等价，去掉重复
    seen = set()
    unique = []
    for ori in orientations:
        key = (round(ori['base_dims'][0], 6), 
               round(ori['base_dims'][1], 6), 
               round(ori['up_dim'], 6))
        if key not in seen:
            seen.add(key)
            unique.append(ori)
    return unique
===
"""
3D装箱系统 — 全局配置与超参数
所有参数均可根据实际场景调整。
"""
import numpy as np

# ============================================================
# 货笼物理尺寸（米） — 超参数
# ============================================================
CAGE_LENGTH = 1.0       # Y方向（里-外，depth）
CAGE_WIDTH  = 0.8       # X方向（左-右，width）
CAGE_HEIGHT = 1.2       # Z方向（高度）

# ============================================================
# 高度图分辨率（米/格）
# ============================================================
HEIGHTMAP_RESOLUTION = 0.01   # 1 cm per cell

# ============================================================
# 货物间隙（米） — 超参数
# ============================================================
PLACEMENT_GAP = 0.01          # 1 cm 间隙，方便实际放置

# ============================================================
# 稳定性阈值
# ============================================================
MIN_SUPPORT_RATIO = 0.60      # 底部至少 60% 面积有支撑
MAX_TILT_ANGLE    = 5.0       # 最大允许倾斜角度（度）
SUPPORT_HEIGHT_TOLERANCE = 0.005  # 支撑高度容差（5mm以内视为有支撑）

# ============================================================
# 评分权重（体现先里后外、先左后右、先下后上）
# ============================================================
WEIGHT_DEPTH = 10.0    # Y方向（越里越好，越大分越高）
WEIGHT_LEFT  = 5.0     # X方向（越左越好）
WEIGHT_LOW   = 8.0     # Z方向（越低越好）
WEIGHT_FIT   = 3.0     # 紧凑度奖励（与已有货物紧贴）

# ============================================================
# 点云预处理参数
# ============================================================
# 体素下采样
VOXEL_DOWNSAMPLE_SIZE = 0.005           # 体素大小（米），0 或 None 表示不下采样

# 离群点移除 — 优先使用 radius outlier removal，备选 SOR
OUTLIER_METHOD = 'radius'               # 'radius' | 'sor'
# Radius outlier removal 参数
RADIUS_OUTLIER_RADIUS    = 0.02         # 搜索半径（米）
RADIUS_OUTLIER_MIN_NEIGHBORS = 6        # 半径内最少点数
# Statistical outlier removal (SOR) 参数
STATISTICAL_OUTLIER_NB_NEIGHBORS = 20   # 统计去噪邻居数
STATISTICAL_OUTLIER_STD_RATIO   = 2.0   # 标准差倍数

# ============================================================
# 高度图聚合方式: 'max' | 'median' | 'p90'
# ============================================================
HEIGHTMAP_AGGREGATION = 'median'        # 默认 median，抗噪性最好

# ============================================================
# 物体坐标系约定
# ============================================================
# 物体在传送带上的初始姿态（自然放置）:
#   物体坐标系原点 = 几何中心
#   物体X轴 = 长度(L)方向
#   物体Y轴 = 宽度(W)方向
#   物体Z轴 = 高度(H)方向（朝上）
#   初始位姿: Roll=0, Pitch=0, Yaw=0

# 6种放置朝向表
# 格式: (底面尺寸在世界X方向, 底面尺寸在世界Y方向, 放置高度, roll, pitch, yaw)
# 世界坐标系: X=笼宽方向, Y=笼深方向, Z=向上
ORIENTATIONS = [
    # ID  底面X    底面Y     up    (roll,       pitch,       yaw)         描述
    #  0  L        W         H     (0,          0,           0)           自然放置
    #  1  W        L         H     (0,          0,           pi/2)        绕Z轴90°
    #  2  L        H         W     (pi/2,       0,           0)           绕X轴90°
    #  3  H        L         W     (pi/2,       0,           pi/2)        绕X轴90° + 绕Z轴90°
    #  4  W        H         L     (0,          -pi/2,       0)           绕Y轴-90° (等价绕Y轴90°使Z朝向-X再调整)
    #  5  H        W         L     (0,          -pi/2,       pi/2)        绕Y轴-90° + 绕Z轴90°
]

def get_orientations(L, W, H):
    """
    给定物体原始尺寸 (L, W, H)，返回所有6种朝向下的参数。
    
    Returns
    -------
    list of dict, 每个dict包含:
        'base_dims': (base_x, base_y)  底面在世界坐标系中的尺寸
        'up_dim': float                放置后的高度
        'roll': float                  弧度
        'pitch': float                 弧度
        'yaw': float                   弧度
        'desc': str                    描述
    """
    orientations = [
        {
            'base_dims': (L, W), 'up_dim': H,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'desc': '自然放置 (LxW底, H高)'
        },
        {
            'base_dims': (W, L), 'up_dim': H,
            'roll': 0.0, 'pitch': 0.0, 'yaw': np.pi / 2,
            'desc': '绕Z轴90° (WxL底, H高)'
        },
        {
            'base_dims': (L, H), 'up_dim': W,
            'roll': np.pi / 2, 'pitch': 0.0, 'yaw': 0.0,
            'desc': '绕X轴90° (LxH底, W高)'
        },
        {
            'base_dims': (H, L), 'up_dim': W,
            'roll': np.pi / 2, 'pitch': 0.0, 'yaw': np.pi / 2,
            'desc': '绕X轴90°+绕Z轴90° (HxL底, W高)'
        },
        {
            'base_dims': (W, H), 'up_dim': L,
            'roll': 0.0, 'pitch': -np.pi / 2, 'yaw': 0.0,
            'desc': '绕Y轴-90° (WxH底, L高)'
        },
        {
            'base_dims': (H, W), 'up_dim': L,
            'roll': 0.0, 'pitch': -np.pi / 2, 'yaw': np.pi / 2,
            'desc': '绕Y轴-90°+绕Z轴90° (HxW底, L高)'
        },
    ]
    
    # 去重：如果L==W或W==H等，某些朝向等价，去掉重复
    seen = set()
    unique = []
    for ori in orientations:
        key = (round(ori['base_dims'][0], 6), 
               round(ori['base_dims'][1], 6), 
               round(ori['up_dim'], 6))
        if key not in seen:
            seen.add(key)
            unique.append(ori)
    return unique
```

- 新增 `VOXEL_DOWNSAMPLE_SIZE = 0.005`（5mm 体素）
- 在预处理流程中，**最先执行** voxel downsample，使点密度均匀、减少后续处理计算量
- 可通过设置为 `0` 或 `None` 禁用

### 2. Radius Outlier Removal

- 新增 `OUTLIER_METHOD` 配置，支持 `'radius'`（默认）和 `'sor'` 两种去噪策略
- **Radius outlier removal** 对"局部高值但不离群"的坏点效果更好，因为它基于固定搜索半径内的邻居数判断
- SOR 作为备选方案保留，通过 `OUTLIER_METHOD = 'sor'` 切换
- 相关参数：`RADIUS_OUTLIER_RADIUS = 0.02`，`RADIUS_OUTLIER_MIN_NEIGHBORS = 6`

### 3. 高度图聚合模式（max / median / p90）

- 新增 `HEIGHTMAP_AGGREGATION` 配置，默认 `'median'`
- `generate_heightmap()` 方法增加 `aggregation` 参数，可按调用覆盖全局设置
- **median** 最抗噪，推荐默认使用；**p90** 平衡噪声与保守估计；**max** 保留原行为

### 4. valid_mask 与有效区域滤波

```diff:point_cloud_processor.py
"""
3D装箱系统 — 点云预处理与高度图生成
"""
import numpy as np
from typing import Tuple, Optional
import scipy.ndimage as ndimage

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from config import (
    HEIGHTMAP_RESOLUTION,
    CAGE_WIDTH, CAGE_LENGTH, CAGE_HEIGHT,
    STATISTICAL_OUTLIER_NB_NEIGHBORS,
    STATISTICAL_OUTLIER_STD_RATIO,
)


class PointCloudProcessor:
    """
    点云预处理器：负责点云去噪、ROI裁剪和高度图生成。
    
    坐标系约定（世界/点云坐标系）:
        X = 笼车宽度方向（左→右）
        Y = 笼车深度方向（外→里）
        Z = 向上
    """
    
    def __init__(self,
                 cage_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 cage_width: float = CAGE_WIDTH,
                 cage_length: float = CAGE_LENGTH,
                 cage_height: float = CAGE_HEIGHT,
                 resolution: float = HEIGHTMAP_RESOLUTION):
        """
        Parameters
        ----------
        cage_origin : (x, y, z)
            笼车内部可用空间的最小角坐标（左-外-底）。
        cage_width : float
            X方向可用宽度（米）。
        cage_length : float
            Y方向可用深度（米）。
        cage_height : float
            Z方向可用高度（米）。
        resolution : float
            高度图分辨率（米/格）。
        """
        self.cage_origin = np.array(cage_origin, dtype=np.float64)
        self.cage_width  = cage_width
        self.cage_length = cage_length
        self.cage_height = cage_height
        self.resolution  = resolution
        
        # 计算高度图尺寸
        self.grid_cols = int(np.ceil(cage_width  / resolution))  # X方向格数
        self.grid_rows = int(np.ceil(cage_length / resolution))  # Y方向格数
        
        # 笼体边界（绝对坐标）
        self.x_min = self.cage_origin[0]
        self.x_max = self.cage_origin[0] + cage_width
        self.y_min = self.cage_origin[1]
        self.y_max = self.cage_origin[1] + cage_length
        self.z_min = self.cage_origin[2]
        self.z_max = self.cage_origin[2] + cage_height

    def preprocess_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        点云预处理：去噪 + ROI裁剪。
        
        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            原始点云。
        
        Returns
        -------
        np.ndarray, shape (M, 3)
            处理后的笼内点云。
        """
        # 1. 统计去噪（如果有Open3D）
        if HAS_OPEN3D and len(points) > STATISTICAL_OUTLIER_NB_NEIGHBORS:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=STATISTICAL_OUTLIER_NB_NEIGHBORS,
                std_ratio=STATISTICAL_OUTLIER_STD_RATIO
            )
            points = np.asarray(pcd.points)
        
        # 2. ROI裁剪：只保留笼内空间
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        )
        return points[mask]

    def generate_heightmap(self, points: np.ndarray) -> np.ndarray:
        """
        从点云生成高度图。每个cell存储该位置最高点的高度。
        
        Parameters
        ----------
        points : np.ndarray, shape (M, 3)
            已预处理的笼内点云。
        
        Returns
        -------
        np.ndarray, shape (grid_rows, grid_cols)
            高度图。值为相对于笼底的高度（米）。
            无点的cell值为 0。
        """
        heightmap = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)
        
        if len(points) == 0:
            return heightmap
        
        # 计算每个点对应的格子索引
        col_indices = ((points[:, 0] - self.x_min) / self.resolution).astype(int)
        row_indices = ((points[:, 1] - self.y_min) / self.resolution).astype(int)
        heights     = points[:, 2] - self.z_min  # 相对于笼底的高度
        
        # 裁剪到有效范围
        valid = (
            (col_indices >= 0) & (col_indices < self.grid_cols) &
            (row_indices >= 0) & (row_indices < self.grid_rows) &
            (heights >= 0)
        )
        col_indices = col_indices[valid]
        row_indices = row_indices[valid]
        heights     = heights[valid]
        
        # 对每个cell取最大高度
        # 使用np.maximum.at进行高效scatter操作
        np.maximum.at(heightmap, (row_indices, col_indices), heights)
        
        # 使用中值滤波消除深度相机的孤立尖刺噪声 (飞点)
        # size=3 表示 3x3 邻域，足以滤除非结构化的强噪点而不破坏货物边缘
        heightmap = ndimage.median_filter(heightmap, size=3)
        
        return heightmap

    def generate_heightmap_from_raw(self, raw_points: np.ndarray) -> np.ndarray:
        """
        一步完成：原始点云 → 预处理 → 高度图。
        """
        processed = self.preprocess_point_cloud(raw_points)
        return self.generate_heightmap(processed)

    def get_heightmap_region(self, heightmap: np.ndarray,
                             row: int, col: int,
                             rows: int, cols: int) -> Optional[np.ndarray]:
        """
        提取高度图上一个矩形区域。
        
        Parameters
        ----------
        heightmap : np.ndarray
            完整高度图。
        row, col : int
            区域左上角（起始行列）。
        rows, cols : int
            区域尺寸（行数、列数）。
        
        Returns
        -------
        np.ndarray or None
            区域子矩阵，若越界返回None。
        """
        if (row < 0 or col < 0 or 
            row + rows > self.grid_rows or 
            col + cols > self.grid_cols):
            return None
        return heightmap[row:row+rows, col:col+cols].copy()

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标 → 高度图格子索引 (row, col)。"""
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """高度图格子索引 → 世界坐标 (x, y)。返回格子中心。"""
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (row + 0.5) * self.resolution
        return x, y
===
"""
3D装箱系统 — 点云预处理与高度图生成

改进:
  1. Voxel downsample 使点密度均匀、提升实时性
  2. 支持 radius outlier removal / SOR 两种去噪策略
  3. 高度图聚合支持 max / median / p90
  4. valid_mask 区分"无点"与"真实低高度"，滤波仅作用于有效区域
"""
import numpy as np
from typing import Tuple, Optional
import scipy.ndimage as ndimage

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from config import (
    HEIGHTMAP_RESOLUTION,
    CAGE_WIDTH, CAGE_LENGTH, CAGE_HEIGHT,
    VOXEL_DOWNSAMPLE_SIZE,
    OUTLIER_METHOD,
    RADIUS_OUTLIER_RADIUS, RADIUS_OUTLIER_MIN_NEIGHBORS,
    STATISTICAL_OUTLIER_NB_NEIGHBORS,
    STATISTICAL_OUTLIER_STD_RATIO,
    HEIGHTMAP_AGGREGATION,
)


class PointCloudProcessor:
    """
    点云预处理器：负责点云去噪、ROI裁剪和高度图生成。
    
    坐标系约定（世界/点云坐标系）:
        X = 笼车宽度方向（左→右）
        Y = 笼车深度方向（外→里）
        Z = 向上
    """
    
    def __init__(self,
                 cage_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 cage_width: float = CAGE_WIDTH,
                 cage_length: float = CAGE_LENGTH,
                 cage_height: float = CAGE_HEIGHT,
                 resolution: float = HEIGHTMAP_RESOLUTION,
                 aggregation: str = HEIGHTMAP_AGGREGATION):
        """
        Parameters
        ----------
        cage_origin : (x, y, z)
            笼车内部可用空间的最小角坐标（左-外-底）。
        cage_width : float
            X方向可用宽度（米）。
        cage_length : float
            Y方向可用深度（米）。
        cage_height : float
            Z方向可用高度（米）。
        resolution : float
            高度图分辨率（米/格）。
        aggregation : str
            高度图聚合方式: 'max' | 'median' | 'p90'。
        """
        self.cage_origin = np.array(cage_origin, dtype=np.float64)
        self.cage_width  = cage_width
        self.cage_length = cage_length
        self.cage_height = cage_height
        self.resolution  = resolution
        self.aggregation = aggregation
        
        # 计算高度图尺寸
        self.grid_cols = int(np.ceil(cage_width  / resolution))  # X方向格数
        self.grid_rows = int(np.ceil(cage_length / resolution))  # Y方向格数
        
        # 笼体边界（绝对坐标）
        self.x_min = self.cage_origin[0]
        self.x_max = self.cage_origin[0] + cage_width
        self.y_min = self.cage_origin[1]
        self.y_max = self.cage_origin[1] + cage_length
        self.z_min = self.cage_origin[2]
        self.z_max = self.cage_origin[2] + cage_height

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------

    def preprocess_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        点云预处理：体素下采样 → 去噪 → ROI裁剪。
        
        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            原始点云。
        
        Returns
        -------
        np.ndarray, shape (M, 3)
            处理后的笼内点云。
        """
        if len(points) == 0:
            return points

        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # ---- 1. Voxel downsample ----
            if VOXEL_DOWNSAMPLE_SIZE and VOXEL_DOWNSAMPLE_SIZE > 0:
                pcd = pcd.voxel_down_sample(voxel_size=VOXEL_DOWNSAMPLE_SIZE)

            # ---- 2. 离群点去除 ----
            pcd = self._remove_outliers(pcd)

            points = np.asarray(pcd.points)
        
        # ---- 3. ROI 裁剪：只保留笼内空间 ----
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        )
        return points[mask]

    def _remove_outliers(self, pcd: 'o3d.geometry.PointCloud') -> 'o3d.geometry.PointCloud':
        """
        根据 OUTLIER_METHOD 选择去噪方式。
        
        - 'radius':  radius outlier removal — 对"局部高值但不离群"的坏点效果更好
        - 'sor':     statistical outlier removal — 经典方式
        """
        n_points = len(pcd.points)
        if n_points < 10:
            return pcd

        if OUTLIER_METHOD == 'radius':
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=RADIUS_OUTLIER_MIN_NEIGHBORS,
                radius=RADIUS_OUTLIER_RADIUS,
            )
        else:
            # SOR
            if n_points > STATISTICAL_OUTLIER_NB_NEIGHBORS:
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=STATISTICAL_OUTLIER_NB_NEIGHBORS,
                    std_ratio=STATISTICAL_OUTLIER_STD_RATIO,
                )
        return pcd

    # ------------------------------------------------------------------
    # 高度图生成
    # ------------------------------------------------------------------

    def generate_heightmap(self,
                           points: np.ndarray,
                           aggregation: Optional[str] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从点云生成高度图 + valid_mask。

        Parameters
        ----------
        points : np.ndarray, shape (M, 3)
            已预处理的笼内点云。
        aggregation : str, optional
            覆盖实例默认聚合方式 ('max' | 'median' | 'p90')。

        Returns
        -------
        heightmap : np.ndarray, shape (grid_rows, grid_cols)
            高度图。值为相对于笼底的高度（米）。
            无点的cell值为 0.0。
        valid_mask : np.ndarray, shape (grid_rows, grid_cols), dtype=bool
            True 表示该 cell 有至少 1 个点投影，False 表示无数据。
        """
        agg = aggregation or self.aggregation
        heightmap = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)
        valid_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)
        
        if len(points) == 0:
            return heightmap, valid_mask
        
        # 计算每个点对应的格子索引
        col_indices = ((points[:, 0] - self.x_min) / self.resolution).astype(int)
        row_indices = ((points[:, 1] - self.y_min) / self.resolution).astype(int)
        heights     = points[:, 2] - self.z_min  # 相对于笼底的高度
        
        # 裁剪到有效范围
        valid = (
            (col_indices >= 0) & (col_indices < self.grid_cols) &
            (row_indices >= 0) & (row_indices < self.grid_rows) &
            (heights >= 0)
        )
        col_indices = col_indices[valid]
        row_indices = row_indices[valid]
        heights     = heights[valid]

        if len(heights) == 0:
            return heightmap, valid_mask

        # ---- 构建 valid_mask ----
        # 任何有点的 cell 标记为有效
        valid_mask[row_indices, col_indices] = True
        
        # ---- 按聚合策略填充高度 ----
        if agg == 'max':
            np.maximum.at(heightmap, (row_indices, col_indices), heights)
        elif agg in ('median', 'p90'):
            # 需要按 cell 分组计算 — 使用 pandas-free 的方式
            # 把 (row, col) 编码为线性索引
            linear_idx = row_indices * self.grid_cols + col_indices
            order = np.argsort(linear_idx)
            linear_idx_sorted = linear_idx[order]
            heights_sorted = heights[order]

            # 找各组的边界
            boundaries = np.flatnonzero(
                np.diff(linear_idx_sorted, prepend=-1)
            )
            # 每组的 cell 索引
            group_keys = linear_idx_sorted[boundaries]
            # 分组结束位置
            boundaries = np.append(boundaries, len(linear_idx_sorted))

            for i in range(len(group_keys)):
                start = boundaries[i]
                end = boundaries[i + 1]
                grp = heights_sorted[start:end]
                r = group_keys[i] // self.grid_cols
                c = group_keys[i] % self.grid_cols
                if agg == 'median':
                    heightmap[r, c] = np.median(grp)
                else:  # p90
                    heightmap[r, c] = np.percentile(grp, 90)
        else:
            raise ValueError(f"Unknown aggregation mode: {agg!r}. Use 'max', 'median', or 'p90'.")

        # ---- 对有效区域做中值滤波（平滑深度相机飞点）----
        # 只对 valid_mask=True 的区域滤波，无数据区域保持 0
        heightmap = self._masked_median_filter(heightmap, valid_mask, size=3)
        
        return heightmap, valid_mask

    @staticmethod
    def _masked_median_filter(heightmap: np.ndarray,
                              valid_mask: np.ndarray,
                              size: int = 3) -> np.ndarray:
        """
        对高度图做中值滤波，但仅在有效区域内操作。
        
        无效(无点)区域在滤波时不参与邻域计算，且滤波后仍保持 0。
        
        实现思路:
          - 将无效区域暂时标记为 NaN
          - 使用 generic_filter + nanmedian 进行滤波
          - 将 NaN 区域恢复为 0
        """
        if not np.any(valid_mask):
            return heightmap

        result = heightmap.copy()
        # 将无效区域设为 NaN，这样 nanmedian 会忽略它们
        result[~valid_mask] = np.nan

        def _nanmedian_func(values):
            valid_vals = values[~np.isnan(values)]
            if len(valid_vals) == 0:
                return np.nan
            return np.median(valid_vals)

        filtered = ndimage.generic_filter(
            result, _nanmedian_func, size=size, mode='constant', cval=np.nan
        )
        
        # 恢复：无效区域保持 0，有效区域使用滤波结果
        out = np.zeros_like(heightmap)
        valid_filtered = valid_mask & ~np.isnan(filtered)
        out[valid_filtered] = filtered[valid_filtered]
        
        return out

    def generate_heightmap_from_raw(self,
                                     raw_points: np.ndarray,
                                     aggregation: Optional[str] = None
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        一步完成：原始点云 → 预处理 → 高度图 + valid_mask。
        """
        processed = self.preprocess_point_cloud(raw_points)
        return self.generate_heightmap(processed, aggregation=aggregation)

    # ------------------------------------------------------------------
    # 兼容接口：返回仅高度图（不含 mask）
    # ------------------------------------------------------------------

    def generate_heightmap_only(self,
                                points: np.ndarray,
                                aggregation: Optional[str] = None) -> np.ndarray:
        """
        与旧版接口兼容：仅返回高度图 ndarray，不返回 valid_mask。
        """
        hm, _ = self.generate_heightmap(points, aggregation=aggregation)
        return hm

    # ------------------------------------------------------------------
    # 区域提取 & 坐标转换
    # ------------------------------------------------------------------

    def get_heightmap_region(self, heightmap: np.ndarray,
                             row: int, col: int,
                             rows: int, cols: int) -> Optional[np.ndarray]:
        """
        提取高度图上一个矩形区域。
        
        Parameters
        ----------
        heightmap : np.ndarray
            完整高度图。
        row, col : int
            区域左上角（起始行列）。
        rows, cols : int
            区域尺寸（行数、列数）。
        
        Returns
        -------
        np.ndarray or None
            区域子矩阵，若越界返回None。
        """
        if (row < 0 or col < 0 or 
            row + rows > self.grid_rows or 
            col + cols > self.grid_cols):
            return None
        return heightmap[row:row+rows, col:col+cols].copy()

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标 → 高度图格子索引 (row, col)。"""
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """高度图格子索引 → 世界坐标 (x, y)。返回格子中心。"""
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (row + 0.5) * self.resolution
        return x, y

```

- `generate_heightmap()` 现在返回 `(heightmap, valid_mask)` 元组
- `valid_mask` 是 bool 数组，`True` = 该 cell 有至少 1 个点投影，`False` = 无数据
- **中值滤波仅在有效区域内操作**：使用 `_masked_median_filter()` 方法，将无效区域设为 NaN，用 `generic_filter + nanmedian` 滤波，最后恢复无效区域为 0
- 这避免了旧版全图 median filter 将"无数据"区域误当作"高度=0"参与滤波的问题

## 文件变更

| 文件 | 变更 |
|------|------|
| [config.py](file:///d:/Code/robot_packing/config.py) | 新增 6 个配置参数 |
| [point_cloud_processor.py](file:///d:/Code/robot_packing/point_cloud_processor.py) | 核心重写：预处理流程 + 高度图生成 + valid_mask |
| [packing_planner.py](file:///d:/Code/robot_packing/packing_planner.py) | 适配新 API（解包 tuple，初始化 valid_mask） |
| [test_packing.py](file:///d:/Code/robot_packing/test_packing.py) | 更新测试：valid_mask 验证 + 聚合模式测试 |

## 兼容性

- 新增 `generate_heightmap_only()` 方法，仅返回 heightmap ndarray，兼容旧调用方式
- `visualize_demo.py` 无需修改（它通过 `planner.heightmap` 访问，不直接调用 `generate_heightmap`）

## 验证

- ✅ 全部单元测试通过（`python test_packing.py`，exit code 0）
