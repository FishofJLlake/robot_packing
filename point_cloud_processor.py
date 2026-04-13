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
    # PLY 文件读取
    # ------------------------------------------------------------------

    @staticmethod
    def load_ply_file(ply_path: str) -> np.ndarray:
        """
        读取 PLY 文件中的点云数据。

        Parameters
        ----------
        ply_path : str
            PLY 文件的路径。

        Returns
        -------
        np.ndarray, shape (N, 3)
            点云坐标 (x, y, z)。

        Raises
        ------
        FileNotFoundError
            文件不存在。
        RuntimeError
            读取失败或文件无有效点。
        """
        import os
        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"PLY 文件不存在: {ply_path}")

        if not HAS_OPEN3D:
            raise RuntimeError("读取 PLY 文件需要 Open3D，请安装: pip install open3d")

        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)

        if len(points) == 0:
            raise RuntimeError(f"PLY 文件无有效点: {ply_path}")

        print(f"  PLY 文件已加载: {ply_path}")
        print(f"  点数: {len(points)}")
        print(f"  范围: X=[{points[:,0].min():.3f}, {points[:,0].max():.3f}], "
              f"Y=[{points[:,1].min():.3f}, {points[:,1].max():.3f}], "
              f"Z=[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

        return points

    def generate_heightmap_from_ply(self,
                                     ply_path: str,
                                     aggregation: Optional[str] = None
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        一步完成：PLY 文件 → 读取 → 预处理 → 高度图 + valid_mask。

        Parameters
        ----------
        ply_path : str
            PLY 文件路径。
        aggregation : str, optional
            覆盖默认聚合方式。

        Returns
        -------
        heightmap : np.ndarray, shape (grid_rows, grid_cols)
        valid_mask : np.ndarray, shape (grid_rows, grid_cols), dtype=bool
        """
        raw_points = self.load_ply_file(ply_path)
        return self.generate_heightmap_from_raw(raw_points, aggregation=aggregation)

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
