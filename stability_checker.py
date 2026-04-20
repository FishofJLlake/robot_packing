"""
3D装箱系统 — 稳定性检测模块

检测放置位置的合法性：
1. 支撑面积比检测
2. 底面倾斜角检测
3. 重心投影稳定性检测（防止倾倒）
"""
import numpy as np
from typing import Tuple
from scipy.spatial import ConvexHull

from config import (
    MIN_SUPPORT_RATIO,
    MAX_TILT_ANGLE,
    SUPPORT_HEIGHT_TOLERANCE,
    HEIGHTMAP_RESOLUTION,
    TRY_PLUS_PACKING,
    MIN_SUPPORT_LENGTH_RATIO_Y,
)


class StabilityChecker:
    """稳定性检测器。"""
    
    def __init__(self,
                 min_support_ratio: float = MIN_SUPPORT_RATIO,
                 max_tilt_angle: float = MAX_TILT_ANGLE,
                 support_height_tol: float = SUPPORT_HEIGHT_TOLERANCE,
                 resolution: float = HEIGHTMAP_RESOLUTION):
        self.min_support_ratio = min_support_ratio
        self.max_tilt_angle_rad = np.deg2rad(max_tilt_angle)
        self.support_height_tol = support_height_tol
        self.resolution = resolution
    
    def check_stability(self, 
                        heightmap_region: np.ndarray,
                        place_height: float,
                        item_base_x: float,
                        item_base_y: float) -> dict:
        """
        综合稳定性检测。
        
        Parameters
        ----------
        heightmap_region : np.ndarray, shape (rows, cols)
            货物底面对应的高度图区域。
        place_height : float
            计划放置的高度（底面高度 = 区域内最大高度值）。
        item_base_x : float
            货物底面X方向尺寸（米）。
        item_base_y : float
            货物底面Y方向尺寸（米）。
        
        Returns
        -------
        dict:
            'is_stable': bool        是否稳定
            'support_ratio': float   支撑面积比
            'tilt_angle': float      底面倾斜角（度）
            'tilt_roll': float       贴合底面的roll微调（弧度）
            'tilt_pitch': float      贴合底面的pitch微调（弧度）
            'reason': str            若不稳定，给出原因
        """
        result = {
            'is_stable': True,
            'support_ratio': 0.0,
            'tilt_angle': 0.0,
            'tilt_roll': 0.0,
            'tilt_pitch': 0.0,
            'reason': '',
            'will_tilt': False
        }
        
        rows, cols = heightmap_region.shape
        total_cells = rows * cols
        
        if total_cells == 0:
            result['is_stable'] = False
            result['reason'] = '底面面积为零'
            return result
        
        # ============================================================
        # 1. 支撑面积比检测
        # ============================================================
        # 放置高度处，底面有支撑的cell（高度差在容差范围内）
        support_mask = (place_height - heightmap_region) <= self.support_height_tol
        # 对于地面（place_height ≈ 0）的情况，地面本身就是支撑
        if place_height < self.support_height_tol:
            support_mask = np.ones_like(heightmap_region, dtype=bool)
        
        support_ratio = np.sum(support_mask) / total_cells
        result['support_ratio'] = support_ratio
        
        if support_ratio < self.min_support_ratio:
            if not TRY_PLUS_PACKING:
                result['is_stable'] = False
                result['reason'] = f'支撑面积不足: {support_ratio:.1%} < {self.min_support_ratio:.1%}'
                return result
            else:
                if support_ratio <= 0.01:
                    result['is_stable'] = False
                    result['reason'] = 'Plus策略: 支撑面积太小(<1%)，极不稳定'
                    return result
                # 检查里外偏心严重时的 Y 向支撑深度率
                y_gradient = self._compute_y_gradient(heightmap_region)
                if y_gradient < -0.01:  # "里侧高外侧低" (里侧支撑，外侧悬空)
                    support_ys, _ = np.where(support_mask)
                    if len(support_ys) > 0:
                        y_min, y_max = np.min(support_ys), np.max(support_ys)
                        support_length_y_ratio = (y_max - y_min + 1) / rows
                        if support_length_y_ratio < MIN_SUPPORT_LENGTH_RATIO_Y:
                            result['is_stable'] = False
                            result['reason'] = f'Plus策略: 里端支撑且外悬空时支撑深度率({support_length_y_ratio:.1%})不足 {MIN_SUPPORT_LENGTH_RATIO_Y:.1%}'
                            return result
        
        # ============================================================
        # 2. 底面倾斜角检测 & 平面拟合
        # ============================================================
        tilt_roll, tilt_pitch, tilt_angle = self._fit_surface_tilt(
            heightmap_region, item_base_x, item_base_y, support_mask
        )
        result['tilt_angle'] = np.rad2deg(tilt_angle)
        result['tilt_roll'] = tilt_roll
        result['tilt_pitch'] = tilt_pitch
        
        if tilt_angle > self.max_tilt_angle_rad:
            result['is_stable'] = False
            result['reason'] = f'底面倾斜过大: {np.rad2deg(tilt_angle):.1f}° > {np.rad2deg(self.max_tilt_angle_rad):.1f}°'
            return result
        
        # ============================================================
        # 3. 重心投影稳定性检测（防倾倒）
        # ============================================================
        # 特别关注：里高外低（Y方向内侧高、外侧低）的情况
        is_cog_stable, cog_reason = self._check_center_of_gravity(
            heightmap_region, support_mask, item_base_x, item_base_y
        )
        if not is_cog_stable:
            if TRY_PLUS_PACKING:
                result['will_tilt'] = True
            else:
                result['is_stable'] = False
                result['reason'] = cog_reason
                return result
        
        return result
    
    def _fit_surface_tilt(self, 
                          heightmap_region: np.ndarray,
                          item_base_x: float,
                          item_base_y: float,
                          support_mask: np.ndarray = None) -> Tuple[float, float, float]:
        """
        用最小二乘法拟合放置区域的平面，计算倾斜角。
        
        拟合平面: z = a*x + b*y + c
        法向量: (-a, -b, 1)  → 归一化后与Z轴的夹角即倾斜角
        
        Returns
        -------
        (tilt_roll, tilt_pitch, tilt_angle) : 弧度
            tilt_roll  = 绕X轴的倾斜（由Y方向高度差引起）
            tilt_pitch = 绕Y轴的倾斜（由X方向高度差引起）
            tilt_angle = 总倾斜角
        """
        rows, cols = heightmap_region.shape
        
        if rows < 2 or cols < 2:
            return 0.0, 0.0, 0.0
        
        # 只拟合有高度值的点
        ys, xs = np.mgrid[0:rows, 0:cols]
        xs_m = xs.ravel() * self.resolution  # 转换为米
        ys_m = ys.ravel() * self.resolution
        zs = heightmap_region.ravel()
        
        # 只使用有实际支撑面的部分来计算倾角（容差之内），避免悬空部分造成数学平面极度倾斜
        if support_mask is not None:
            mask = support_mask.ravel()
        else:
            mask = zs > 0
            
        if np.sum(mask) < 3:
            return 0.0, 0.0, 0.0
        
        xs_fit = xs_m[mask]
        ys_fit = ys_m[mask]
        zs_fit = zs[mask]
        
        # 最小二乘拟合: z = a*x + b*y + c
        A = np.column_stack([xs_fit, ys_fit, np.ones(len(xs_fit))])
        try:
            result = np.linalg.lstsq(A, zs_fit, rcond=None)
            coeffs = result[0]
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0
        
        a, b, c = coeffs
        
        # 法向量: n = (-a, -b, 1), 归一化
        normal = np.array([-a, -b, 1.0])
        normal /= np.linalg.norm(normal)
        
        # 总倾斜角 = 法向量与Z轴的夹角
        tilt_angle = np.arccos(np.clip(abs(normal[2]), 0, 1))
        
        # 分解为 roll (绕X) 和 pitch (绕Y)
        tilt_pitch = np.arctan2(-a, 1.0)  # X方向梯度 → 绕Y轴倾斜
        tilt_roll  = np.arctan2(b, 1.0)   # Y方向梯度 → 绕X轴倾斜（注意方向）
        
        return tilt_roll, tilt_pitch, tilt_angle
    
    def _check_center_of_gravity(self,
                                  heightmap_region: np.ndarray,
                                  support_mask: np.ndarray,
                                  item_base_x: float,
                                  item_base_y: float) -> Tuple[bool, str]:
        """
        检查重心投影是否在支撑面内。
        
        重点：当底面"里高外低"时（Y方向梯度为负），
        货物重心会向外侧偏移，可能倾倒出笼。
        
        Returns
        -------
        (is_stable, reason) : (bool, str)
        """
        rows, cols = heightmap_region.shape
        
        # 获取有支撑的cell的坐标
        support_ys, support_xs = np.where(support_mask)
        
        if len(support_xs) < 3:
            return False, '支撑点过少，无法形成稳定支撑面'
        
        # 支撑点坐标转换为米（相对于物体底面左下角）
        support_points_x = support_xs * self.resolution
        support_points_y = support_ys * self.resolution
        
        # 货物重心投影（物体底面中心）
        cog_x = item_base_x / 2.0
        cog_y = item_base_y / 2.0
        
        # 计算支撑面凸包
        try:
            points_2d = np.column_stack([support_points_x, support_points_y])
            # 去重以避免退化的凸包
            points_2d_unique = np.unique(points_2d, axis=0)
            
            if len(points_2d_unique) < 3:
                # 不足以形成凸包，检查是否至少在一条线上两侧有支撑
                return self._check_linear_support(
                    points_2d_unique, cog_x, cog_y, item_base_x, item_base_y
                )
            
            hull = ConvexHull(points_2d_unique)
            
            # 检查重心是否在凸包内
            # 使用射线法
            if not self._point_in_convex_hull(cog_x, cog_y, points_2d_unique[hull.vertices]):
                return False, '重心投影落在支撑面凸包外，有倾倒风险'
            
            # 计算重心到凸包边缘的最小距离（安全裕度）
            min_dist = self._point_to_hull_distance(
                cog_x, cog_y, points_2d_unique[hull.vertices]
            )
            
            # 安全裕度：重心距凸包边缘至少 1cm 或 底面尺寸的 5%
            safety_margin = max(0.01, min(item_base_x, item_base_y) * 0.05)
            
            if min_dist < safety_margin:
                # 检查倾斜方向——特别是"里高外低"时
                y_gradient = self._compute_y_gradient(heightmap_region)
                if y_gradient < -0.02:  # 里高外低（Y正=里，梯度为负说明从里到外高度降低）
                    return False, f'底面里高外低(梯度={y_gradient:.3f})，重心距支撑边缘仅{min_dist*100:.1f}cm，有向外倾倒风险'
            
        except Exception:
            # 凸包计算失败时，使用简单的包围盒检测
            x_min_s, x_max_s = np.min(support_points_x), np.max(support_points_x)
            y_min_s, y_max_s = np.min(support_points_y), np.max(support_points_y)
            
            if not (x_min_s < cog_x < x_max_s and y_min_s < cog_y < y_max_s):
                return False, '重心投影落在支撑区域外'
        
        return True, ''
    
    def _check_linear_support(self, points, cog_x, cog_y, base_x, base_y):
        """当支撑点不足以形成凸包时的退化检测。"""
        if len(points) == 0:
            return False, '无支撑点'
        if len(points) == 1:
            return False, '仅一个支撑区域，不稳定'
        # 两个支撑区域：检查重心是否在连线附近
        mid_x = np.mean(points[:, 0])
        mid_y = np.mean(points[:, 1])
        dist = np.sqrt((cog_x - mid_x)**2 + (cog_y - mid_y)**2)
        max_dist = np.sqrt(base_x**2 + base_y**2) * 0.1
        if dist > max_dist:
            return False, '支撑区域不足以稳定支撑重心'
        return True, ''
    
    @staticmethod
    def _point_in_convex_hull(px, py, hull_vertices):
        """
        判断点(px,py)是否在凸多边形内（射线交叉法）。
        hull_vertices: (K, 2) 凸包顶点，按顺序排列。
        """
        n = len(hull_vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = hull_vertices[i]
            xj, yj = hull_vertices[j]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside
    
    @staticmethod
    def _point_to_hull_distance(px, py, hull_vertices):
        """计算点到凸包边缘的最小距离。"""
        n = len(hull_vertices)
        min_dist = float('inf')
        for i in range(n):
            x1, y1 = hull_vertices[i]
            x2, y2 = hull_vertices[(i + 1) % n]
            
            # 点到线段的距离
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                dist = np.sqrt((px - x1)**2 + (py - y1)**2)
            else:
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
            min_dist = min(min_dist, dist)
        return min_dist
    
    def _compute_y_gradient(self, heightmap_region: np.ndarray) -> float:
        """
        计算高度图区域沿Y方向（里→外）的平均梯度。
        正值：外高里低；负值：里高外低。
        """
        rows, cols = heightmap_region.shape
        if rows < 2:
            return 0.0
        
        # Y方向从小(外)到大(里)，计算上半部分与下半部分的平均高度差
        mid = rows // 2
        inner_mean = np.mean(heightmap_region[mid:, :])   # 里侧（大Y）
        outer_mean = np.mean(heightmap_region[:mid, :])    # 外侧（小Y）
        
        # 梯度 = (外 - 里) / 距离
        distance = (rows / 2) * self.resolution
        if distance < 1e-6:
            return 0.0
        return (outer_mean - inner_mean) / distance
