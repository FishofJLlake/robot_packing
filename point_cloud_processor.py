"""
3D装箱系统 — 点云预处理与高度图生成

改进:
  1. Voxel downsample 使点密度均匀、提升实时性
  2. 支持 radius outlier removal / SOR 两种去噪策略
  3. 高度图聚合支持 max / median / p90
  4. valid_mask 区分"无点"与"真实低高度"，滤波仅作用于有效区域
"""
from collections import deque

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
    PLY_CROP_X, PLY_CROP_Y, PLY_CROP_Z,
    PLANE_HEIGHT_DIFF_THRESHOLD,
    PLANE_SMALL_REGION_MAX_CELLS,
    PLANE_FIT_REPLACE_MAX_TILT_DEG,
    PLANE_FIT_MAX_RMSE,
    PLANE_FIT_MIN_CELLS,
    PLANE_FIT_SLOPE_BLEND_ALPHA,
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
        
        # 记录最新处理的点云数据用于外部3D可视化叠加
        self.latest_points: Optional[np.ndarray] = None
        self.latest_colors: Optional[np.ndarray] = None
        self.latest_raw_heightmap: Optional[np.ndarray] = None
        self.latest_fitted_heightmap: Optional[np.ndarray] = None
        self.latest_plane_label_map: Optional[np.ndarray] = None
        
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

    def preprocess_point_cloud(self, points: np.ndarray, is_real: bool = False, colors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预处理点云：下采样、去噪、裁剪。

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            原始点云。
        is_real : bool, optional
            是否为真实采集点云（影响裁剪策略）。
        colors : np.ndarray, optional
            点云颜色。
        
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            处理后的有效点云及对应颜色。
        """
        if len(points) == 0:
            return points

        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # ---- 1. Voxel downsample ----
            if VOXEL_DOWNSAMPLE_SIZE and VOXEL_DOWNSAMPLE_SIZE > 0:
                pcd = pcd.voxel_down_sample(voxel_size=VOXEL_DOWNSAMPLE_SIZE)

            # ---- 2. 离群点去除 ----
            pcd = self._remove_outliers(pcd)

            points = np.asarray(pcd.points)
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
        
        # ---- 3. ROI 裁剪：提取有效空间区域 ----
        # 融合逻辑：真实点云按xyz配置裁剪，仿真点云按笼子范围裁剪
        if is_real:
            # 记录裁剪前的范围给用户反馈
            if len(points) > 0:
                print(f"  [信息] 裁剪前范围: X=[{points[:,0].min():.1f}, {points[:,0].max():.1f}], "
                      f"Y=[{points[:,1].min():.1f}, {points[:,1].max():.1f}], "
                      f"Z=[{points[:,2].min():.1f}, {points[:,2].max():.1f}]")
            
            mask = np.ones(len(points), dtype=bool)
            if PLY_CROP_X is not None: mask &= (points[:, 0] >= PLY_CROP_X[0]) & (points[:, 0] <= PLY_CROP_X[1])
            if PLY_CROP_Y is not None: mask &= (points[:, 1] >= PLY_CROP_Y[0]) & (points[:, 1] <= PLY_CROP_Y[1])
            if PLY_CROP_Z is not None: mask &= (points[:, 2] >= PLY_CROP_Z[0]) & (points[:, 2] <= PLY_CROP_Z[1])
            
            # 反馈裁剪参数
            print(f"  [信息] 实际采用裁剪配置: X={PLY_CROP_X}, Y={PLY_CROP_Y}, Z={PLY_CROP_Z}")
        else:
            mask = (
                (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
                (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
                (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
            )
            
        pts_cropped = points[mask]
        if is_real and len(pts_cropped) > 0:
            print(f"  [信息] 裁剪后点数: {len(points)} -> {len(pts_cropped)}")
            print(f"  [信息] 裁剪后范围: X=[{pts_cropped[:,0].min():.1f}, {pts_cropped[:,0].max():.1f}], "
                  f"Y=[{pts_cropped[:,1].min():.1f}, {pts_cropped[:,1].max():.1f}], "
                  f"Z=[{pts_cropped[:,2].min():.1f}, {pts_cropped[:,2].max():.1f}]")
        
        return pts_cropped, (colors[mask] if colors is not None else None)

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
                           is_real: bool = False,
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
        
        # 针对真实点云：装箱区域不再是固定全笼大小，而是根据 PLY_CROP 的设定动态圈定
        if is_real:
            import math
            from config import PLY_CROP_X, PLY_CROP_Y, PLY_CROP_Z
            
            # 使用用户定义的边界作为高度图的严格物理原点和范围
            x_min_real = PLY_CROP_X[0] if (PLY_CROP_X and not math.isinf(PLY_CROP_X[0])) else np.min(points[:, 0]) if len(points) > 0 else 0
            y_min_real = PLY_CROP_Y[0] if (PLY_CROP_Y and not math.isinf(PLY_CROP_Y[0])) else np.min(points[:, 1]) if len(points) > 0 else 0
            z_min_real = PLY_CROP_Z[0] if (PLY_CROP_Z and not math.isinf(PLY_CROP_Z[0])) else np.min(points[:, 2]) if len(points) > 0 else 0
            
            x_max_real = PLY_CROP_X[1] if (PLY_CROP_X and not math.isinf(PLY_CROP_X[1])) else np.max(points[:, 0]) if len(points) > 0 else 0
            y_max_real = PLY_CROP_Y[1] if (PLY_CROP_Y and not math.isinf(PLY_CROP_Y[1])) else np.max(points[:, 1]) if len(points) > 0 else 0
            z_max_real = PLY_CROP_Z[1] if (PLY_CROP_Z and not math.isinf(PLY_CROP_Z[1])) else np.max(points[:, 2]) if len(points) > 0 else 0

            self.real_scale = 0.001 if (x_max_real - x_min_real > 10.0 or abs(x_min_real) > 10.0) else 1.0

            # 更新网格全局尺寸（非常重要，规划器将自动按这些新尺寸执行装箱约束）
            real_width = (x_max_real - x_min_real) * self.real_scale
            real_length = (y_max_real - y_min_real) * self.real_scale
            
            self.cage_width = real_width
            self.cage_length = real_length
            self.cage_height = (z_max_real - z_min_real) * self.real_scale

            self.grid_cols = max(1, int(np.ceil(real_width / self.resolution)))
            self.grid_rows = max(1, int(np.ceil(real_length / self.resolution)))

            self.x_min_real, self.y_min_real, self.z_min_real = x_min_real, y_min_real, z_min_real
        else:
            # 仿真点云：始终按照初始化时的满载物理笼子规格计算网格
            self.grid_cols = max(1, int(np.ceil((self.x_max - self.x_min) / self.resolution)))
            self.grid_rows = max(1, int(np.ceil((self.y_max - self.y_min) / self.resolution)))

        heightmap = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)
        valid_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)
        
        if len(points) == 0:
            return heightmap, valid_mask
        
        # 计算每个点对应的格子索引
        # 融合逻辑：真实点云需要动态对齐到矩阵原点并做单位换算 (如从毫米转为米)
        if is_real:
            col_indices = ((points[:, 0] - self.x_min_real) * self.real_scale / self.resolution).astype(int)
            row_indices = ((points[:, 1] - self.y_min_real) * self.real_scale / self.resolution).astype(int)
            heights     = (points[:, 2] - self.z_min_real) * self.real_scale
        else:
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
        # heightmap = self._masked_median_filter(heightmap, valid_mask, size=3)
        
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

    def _fit_local_planes_from_heightmap(self,
                                         heightmap: np.ndarray,
                                         valid_mask: np.ndarray
                                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit local planes on the whole heightmap and return a smoother planning
        heightmap plus a plane label map.
        """
        fitted_heightmap = heightmap.copy()
        plane_label_map = np.full(heightmap.shape, -1, dtype=np.int32)

        if not np.any(valid_mask):
            return fitted_heightmap, plane_label_map

        assigned_mask = np.zeros_like(valid_mask, dtype=bool)
        tolerance = float(PLANE_HEIGHT_DIFF_THRESHOLD)
        max_tilt_deg = float(PLANE_FIT_REPLACE_MAX_TILT_DEG)
        max_rmse = float(PLANE_FIT_MAX_RMSE)
        min_cells = int(PLANE_FIT_MIN_CELLS)
        slope_blend_alpha = float(np.clip(PLANE_FIT_SLOPE_BLEND_ALPHA, 0.0, 1.0))
        plane_id = 0

        for row in range(heightmap.shape[0]):
            for col in range(heightmap.shape[1]):
                if not valid_mask[row, col] or assigned_mask[row, col]:
                    continue

                component = self._collect_plane_component(
                    heightmap, valid_mask, assigned_mask, row, col, tolerance
                )
                if not component:
                    continue

                comp_rows = np.array([r for r, _ in component], dtype=np.int32)
                comp_cols = np.array([c for _, c in component], dtype=np.int32)

                fit_result = self._fit_plane_component_with_metrics(
                    heightmap, comp_rows, comp_cols
                )
                if fit_result is None:
                    assigned_mask[comp_rows, comp_cols] = True
                    continue
                fitted_values, _, tilt_deg, rmse = fit_result

                residuals = np.abs(fitted_values - heightmap[comp_rows, comp_cols])
                inlier_mask = residuals <= tolerance
                is_fit_candidate = (
                    comp_rows.size >= min_cells and
                    rmse <= max_rmse
                )

                if is_fit_candidate and np.sum(inlier_mask) >= 3:
                    inlier_rows = comp_rows[inlier_mask]
                    inlier_cols = comp_cols[inlier_mask]
                    plane_label_map[inlier_rows, inlier_cols] = plane_id

                    if tilt_deg <= max_tilt_deg:
                        # 低倾角：按拟合值替换，优先去噪和平整局部小波纹
                        fitted_heightmap[inlier_rows, inlier_cols] = fitted_values[inlier_mask]
                    else:
                        # 高倾角：采用弱拟合，保留原始渐变趋势，避免过度拉平
                        raw_vals = heightmap[inlier_rows, inlier_cols]
                        fitted_heightmap[inlier_rows, inlier_cols] = (
                            (1.0 - slope_blend_alpha) * raw_vals +
                            slope_blend_alpha * fitted_values[inlier_mask]
                        )
                    assigned_mask[inlier_rows, inlier_cols] = True
                    plane_id += 1
                else:
                    # Rejected fit candidate: keep original heights unchanged.
                    # Skip plane labels here to avoid downstream merge rewriting.
                    assigned_mask[comp_rows, comp_cols] = True

        fitted_heightmap, plane_label_map = self._merge_small_plane_regions(
            fitted_heightmap, plane_label_map, valid_mask
        )
        return fitted_heightmap, plane_label_map

    @staticmethod
    def _collect_plane_component(heightmap: np.ndarray,
                                 valid_mask: np.ndarray,
                                 assigned_mask: np.ndarray,
                                 start_row: int,
                                 start_col: int,
                                 tolerance: float):
        """Collect one local height-consistent component."""
        component = []
        seen = np.zeros_like(valid_mask, dtype=bool)
        queue = deque([(start_row, start_col)])
        seen[start_row, start_col] = True

        while queue:
            row, col = queue.popleft()
            component.append((row, col))
            base_height = heightmap[row, col]

            for d_row, d_col in (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ):
                next_row = row + d_row
                next_col = col + d_col

                if (next_row < 0 or next_row >= heightmap.shape[0] or
                        next_col < 0 or next_col >= heightmap.shape[1]):
                    continue
                if seen[next_row, next_col] or assigned_mask[next_row, next_col]:
                    continue
                if not valid_mask[next_row, next_col]:
                    continue
                if abs(heightmap[next_row, next_col] - base_height) > tolerance:
                    continue

                seen[next_row, next_col] = True
                queue.append((next_row, next_col))

        return component

    def _fit_plane_component(self,
                             heightmap: np.ndarray,
                             comp_rows: np.ndarray,
                             comp_cols: np.ndarray) -> Optional[np.ndarray]:
        """Fit z = ax + by + c for one connected component."""
        fit_result = self._fit_plane_component_with_metrics(heightmap, comp_rows, comp_cols)
        if fit_result is None:
            return None
        fitted_values, _, _, _ = fit_result
        return fitted_values

    def _fit_plane_component_with_metrics(self,
                                          heightmap: np.ndarray,
                                          comp_rows: np.ndarray,
                                          comp_cols: np.ndarray
                                          ) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        """
        Fit z = ax + by + c for one connected component.

        Returns
        -------
        (fitted_values, coeffs, tilt_deg, rmse) or None
        """
        if comp_rows.size < 3:
            return None

        xs = comp_cols.astype(np.float64) * self.resolution
        ys = comp_rows.astype(np.float64) * self.resolution
        zs = heightmap[comp_rows, comp_cols]
        design = np.column_stack([xs, ys, np.ones_like(xs)])

        try:
            coeffs, _, rank, _ = np.linalg.lstsq(design, zs, rcond=None)
        except np.linalg.LinAlgError:
            return None

        if rank < 3:
            return None

        fitted_values = design @ coeffs
        residuals = fitted_values - zs
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        a, b = float(coeffs[0]), float(coeffs[1])
        tilt_rad = np.arctan(np.sqrt(a * a + b * b))
        tilt_deg = float(np.rad2deg(tilt_rad))

        return fitted_values, coeffs, tilt_deg, rmse

    def _merge_small_plane_regions(self,
                                   fitted_heightmap: np.ndarray,
                                   plane_label_map: np.ndarray,
                                   valid_mask: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge very small isolated plane regions into surrounding larger planes.
        """
        max_small_region_cells = int(PLANE_SMALL_REGION_MAX_CELLS)
        if max_small_region_cells <= 0:
            return fitted_heightmap, plane_label_map

        labels = plane_label_map[plane_label_map >= 0]
        if labels.size == 0:
            return fitted_heightmap, plane_label_map

        label_counts = {
            int(label): int(np.sum(plane_label_map == label))
            for label in np.unique(labels)
        }

        for label, count in sorted(label_counts.items(), key=lambda item: item[1]):
            if count > max_small_region_cells:
                continue

            region_rows, region_cols = np.where(plane_label_map == label)
            if region_rows.size == 0:
                continue

            neighbor_votes = {}
            for row, col in zip(region_rows, region_cols):
                for d_row, d_col in (
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)
                ):
                    next_row = row + d_row
                    next_col = col + d_col
                    if (next_row < 0 or next_row >= plane_label_map.shape[0] or
                            next_col < 0 or next_col >= plane_label_map.shape[1]):
                        continue
                    if not valid_mask[next_row, next_col]:
                        continue

                    neighbor_label = int(plane_label_map[next_row, next_col])
                    if neighbor_label < 0 or neighbor_label == label:
                        continue

                    neighbor_votes[neighbor_label] = neighbor_votes.get(neighbor_label, 0) + 1

            if not neighbor_votes:
                continue

            target_label = max(
                neighbor_votes,
                key=lambda candidate: (neighbor_votes[candidate], label_counts.get(candidate, 0))
            )
            target_rows, target_cols = np.where(plane_label_map == target_label)
            target_fit = self._fit_plane_component(
                fitted_heightmap, target_rows.astype(np.int32), target_cols.astype(np.int32)
            )
            if target_fit is not None:
                xs = region_cols.astype(np.float64) * self.resolution
                ys = region_rows.astype(np.float64) * self.resolution
                design = np.column_stack([xs, ys, np.ones_like(xs)])
                coeffs, _, rank, _ = np.linalg.lstsq(
                    np.column_stack([
                        target_cols.astype(np.float64) * self.resolution,
                        target_rows.astype(np.float64) * self.resolution,
                        np.ones_like(target_rows, dtype=np.float64),
                    ]),
                    fitted_heightmap[target_rows, target_cols],
                    rcond=None,
                )
                if rank >= 3:
                    fitted_heightmap[region_rows, region_cols] = design @ coeffs
                else:
                    fitted_heightmap[region_rows, region_cols] = np.median(
                        fitted_heightmap[target_rows, target_cols]
                    )
            else:
                fitted_heightmap[region_rows, region_cols] = np.median(
                    fitted_heightmap[target_rows, target_cols]
                )

            plane_label_map[region_rows, region_cols] = target_label
            label_counts[target_label] = label_counts.get(target_label, 0) + count
            label_counts[label] = 0

        return fitted_heightmap, plane_label_map

    def _cache_heightmap_artifacts(self,
                                   raw_heightmap: np.ndarray,
                                   fitted_heightmap: np.ndarray,
                                   plane_label_map: np.ndarray):
        """Cache raw/fitted heightmap artifacts for downstream modules."""
        self.latest_raw_heightmap = raw_heightmap.copy()
        self.latest_fitted_heightmap = fitted_heightmap.copy()
        self.latest_plane_label_map = plane_label_map.copy()

    def generate_heightmap_from_raw(self,
                                     raw_points: np.ndarray,
                                     is_real: bool = False,
                                     aggregation: Optional[str] = None
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        一步完成：点云数据 → 预处理(融合裁剪/去噪) → 对齐高度图 + valid_mask。
        """
        processed, _ = self.preprocess_point_cloud(raw_points, is_real=is_real)
        raw_heightmap, valid_mask = self.generate_heightmap(
            processed, is_real=is_real, aggregation=aggregation
        )
        fitted_heightmap, plane_label_map = self._fit_local_planes_from_heightmap(
            raw_heightmap, valid_mask
        )
        self._cache_heightmap_artifacts(raw_heightmap, fitted_heightmap, plane_label_map)
        return fitted_heightmap, valid_mask

    # ------------------------------------------------------------------
    # PLY 文件读取
    # ------------------------------------------------------------------

    @staticmethod
    def _visualize_with_axes(pcd, title="Point Cloud"):
        """尝试使用带数值网格轴的高级视图，若不可用则回退至经典视图加坐标系显示。"""
        import open3d as o3d
        import sys
        
        print(f"  >>> [可视化] 正在弹窗显示 {title}。请关闭弹窗后继续运行...")
        try:
            # 尝试使用 Open3D 高级 GUI (支持坐标系数值网格，体验最佳)
            o3d.visualization.draw(
                {"name": "pcd", "geometry": pcd},
                title=title,
                show_skybox=False
            )
        except Exception:
            # 如果高级 GUI 不支持(例如系统无特定共享库)，回退到经典带有坐标基准(红=X, 绿=Y, 蓝=Z)的显示
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, frame], window_name=title)

    @staticmethod
    def load_ply_file(ply_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        纯净提取 PLY 文件中的点云坐标与颜色。

        Parameters
        ----------
        ply_path : str
            PLY 文件的路径。

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            点云坐标 (x, y, z) 和 颜色。
        """
        import os
        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"PLY 文件不存在: {ply_path}")

        if not HAS_OPEN3D:
            raise RuntimeError("读取 PLY 文件需要 Open3D，请安装: pip install open3d")

        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        if len(points) == 0:
            raise RuntimeError(f"PLY 文件无有效点: {ply_path}")
            
        # =========================================================
        # 坐标系转换 (根据现场视觉基准与系统约定的映射关系)
        # 原始坐标：X=向右, Y=向上, Z=垂直向外
        # 目标装箱系统约定坐标：X=向右, Y=向里, Z=向上
        # =========================================================
        # converted_points = np.empty_like(points)
        # converted_points[:, 0] = points[:, 0]     # X 保持不变 (向右)
        # converted_points[:, 1] = -points[:, 2]    # Y 变成向里 (原Z向外的反方向)
        # converted_points[:, 2] = points[:, 1]     # Z 变成向上 (原Y向上的方向)

        return points, colors

    def generate_heightmap_from_ply(self,
                                     ply_path: str,
                                     aggregation: Optional[str] = None
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        统一完成真实场景的预演与处理：读取 → 预可视化 → 裁剪去噪 → 结果可视化 → 对齐高度图。
        """
        import os
        raw_points, raw_colors = self.load_ply_file(ply_path)
        
        if HAS_OPEN3D:
            pcd_raw = o3d.geometry.PointCloud()
            pcd_raw.points = o3d.utility.Vector3dVector(raw_points)
            if raw_colors is not None:
                pcd_raw.colors = o3d.utility.Vector3dVector(raw_colors)
            PointCloudProcessor._visualize_with_axes(pcd_raw, title=f"Raw PLY - {os.path.basename(ply_path)}")
            
        # 统一使用预处理管线
        processed_pts, processed_cls = self.preprocess_point_cloud(raw_points, is_real=True, colors=raw_colors)
        
        if HAS_OPEN3D and len(processed_pts) > 0:
            pcd_proc = o3d.geometry.PointCloud()
            pcd_proc.points = o3d.utility.Vector3dVector(processed_pts)
            if processed_cls is not None:
                pcd_proc.colors = o3d.utility.Vector3dVector(processed_cls)
            PointCloudProcessor._visualize_with_axes(pcd_proc, title=f"Cropped & Processed PLY - {os.path.basename(ply_path)}")
            
        # 生成高度图 (这步同时会锁定新装箱原点、缩放配置)
        raw_heightmap, valid_mask = self.generate_heightmap(
            processed_pts, is_real=True, aggregation=aggregation
        )
        fitted_heightmap, plane_label_map = self._fit_local_planes_from_heightmap(
            raw_heightmap, valid_mask
        )
        self._cache_heightmap_artifacts(raw_heightmap, fitted_heightmap, plane_label_map)
        
        # 将发往 3D 结果总视图的散点坐标进行刚体平移对齐，与规划器里的笼子(0,0,0)原点相认
        aligned_pts = np.copy(processed_pts)
        if hasattr(self, 'x_min_real'):
            aligned_pts[:, 0] = (aligned_pts[:, 0] - self.x_min_real) * getattr(self, 'real_scale', 1.0)
            aligned_pts[:, 1] = (aligned_pts[:, 1] - getattr(self, 'y_min_real', 0.0)) * getattr(self, 'real_scale', 1.0)
            aligned_pts[:, 2] = (aligned_pts[:, 2] - getattr(self, 'z_min_real', 0.0)) * getattr(self, 'real_scale', 1.0)
            
        # 记录映射对齐后的结果供外部工具使用（如3D装箱可视化）
        self.latest_points = aligned_pts
        self.latest_colors = processed_cls
        
        return fitted_heightmap, valid_mask

    # ------------------------------------------------------------------
    # 兼容接口：返回仅高度图（不含 mask）
    # ------------------------------------------------------------------

    def generate_heightmap_only(self,
                                points: np.ndarray,
                                aggregation: Optional[str] = None) -> np.ndarray:
        """
        与旧版接口兼容：仅返回高度图 ndarray，不返回 valid_mask。
        """
        hm, _ = self.generate_heightmap(points, is_real=False, aggregation=aggregation)
        return hm

    # ------------------------------------------------------------------
    # 工具函数：真实空间逆变换 (供给下游机械臂)
    # ------------------------------------------------------------------
    
    def to_camera_absolute_pose(self, x_planner: float, y_planner: float, z_planner: float, rot_matrix_pkg: np.ndarray):
        """
        将基于本地局部特征装箱坐标系 (X向右, Y向里, Z向上) 算出的 6D位姿，
        完全逆变换倒推回真实深度相机原始视角的坐标系 (X向右, Y向上, Z向外) 的绝对物理刻度。
        
        Returns
        -------
        (abs_x, abs_y, abs_z) : Tuple[float, float, float]
            绝对物理位置 (单位对应于传入底层相机时使用的单位，若原始PLY用的是米，则此处为米)
        rot_matrix_cam : np.ndarray
            基于相机坐标系的旋转矩阵
        euler_xyz_deg : Tuple[float, float, float]
            基于相机的欧拉角 (Roll, Pitch, Yaw)
        quat_xyzw : Tuple[float, float, float, float]
            基于相机的四元数
        """
        import scipy.spatial.transform as transform
        scale = getattr(self, 'real_scale', 1.0)
        x_min = getattr(self, 'x_min_real', 0.0)
        y_min = getattr(self, 'y_min_real', 0.0)
        z_min = getattr(self, 'z_min_real', 0.0)
        
        # 1. 解除尺度与本地偏移，恢复到 Pkg 视角的绝对物理尺码
        x_pkg = x_planner / scale + x_min
        y_pkg = y_planner / scale + y_min
        z_pkg = z_planner / scale + z_min
        
        # 2. 从 Pkg 坐标系倒推回 Cam 坐标系
        # 之前前置转换是: X_pkg = X_cam, Y_pkg = -Z_cam, Z_pkg = Y_cam
        # 现在反推还原: X_cam = X_pkg, Z_cam = -Y_pkg, Y_cam = Z_pkg
        # x_cam = x_pkg
        # y_cam = z_pkg
        # z_cam = -y_pkg
        
        # 3. 倒推旋转矩阵
        # 让相机向下的矩阵变换回到相机的局部基准
        R_cp = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ])
        rot_matrix_cam = R_cp @ rot_matrix_pkg
        
        try:
            r = transform.Rotation.from_matrix(rot_matrix_cam)
            abs_euler = tuple(r.as_euler('xyz', degrees=True))
            abs_quat = tuple(r.as_quat())  # (x, y, z, w)
        except Exception:
            abs_euler = (0.0, 0.0, 0.0)
            abs_quat = (0.0, 0.0, 0.0, 1.0)
            
        return (x_pkg, y_pkg, z_pkg), rot_matrix_cam, abs_euler, abs_quat


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
