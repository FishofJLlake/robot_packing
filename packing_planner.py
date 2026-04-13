"""
3D装箱系统 — 核心装箱规划器

实现基于高度图的在线装箱算法，支持多种朝向，
评分规则为：先里后外、先左后右、先下后上。

增强功能:
  1. XY-only旋转开关
  2. 外侧高度约束（比例检查，忽略空隙）
  3. 多维度评分策略（平整度、层完成度、空隙惩罚、阶梯奖励）
"""
import numpy as np
from typing import List, Optional, Tuple
import math

from config import (
    CAGE_WIDTH, CAGE_LENGTH, CAGE_HEIGHT,
    HEIGHTMAP_RESOLUTION,
    PLACEMENT_GAP,
    WEIGHT_DEPTH, WEIGHT_LEFT, WEIGHT_LOW, WEIGHT_FIT,
    WEIGHT_SURFACE_FLAT, WEIGHT_LAYER_COMPLETE,
    WEIGHT_VOID_PENALTY, WEIGHT_STEP_PROFILE,
    XY_ONLY_ROTATION,
    OUTER_HEIGHT_TOLERANCE, OUTER_HEIGHT_CHECK_RATIO,
    get_orientations,
)
from point_cloud_processor import PointCloudProcessor
from stability_checker import StabilityChecker
from pose_generator import compute_6d_pose, format_pose_string


class PackingPlanner:
    """
    装箱规划器。
    
    给定当前笼体高度图和待放置货物尺寸，
    计算最优放置位姿（6D Pose）。
    """
    
    def __init__(self,
                 processor: PointCloudProcessor,
                 stability_checker: Optional[StabilityChecker] = None,
                 xy_only: bool = XY_ONLY_ROTATION,
                 outer_height_tol: float = OUTER_HEIGHT_TOLERANCE,
                 outer_height_ratio: float = OUTER_HEIGHT_CHECK_RATIO):
        """
        Parameters
        ----------
        processor : PointCloudProcessor
            点云处理器（持有笼体参数和高度图生成能力）。
        stability_checker : StabilityChecker, optional
            稳定性检测器，若不提供则使用默认参数创建。
        xy_only : bool
            仅XY旋转模式（只保持底面朝下，绕Z轴旋转0°/90°）。
        outer_height_tol : float
            外侧高度约束阈值（米），默认0.05即5cm。
        outer_height_ratio : float
            外侧违规列占有效列的比例阈值，超过则拒绝。
        """
        self.processor = processor
        self.checker = stability_checker or StabilityChecker()
        self.xy_only = xy_only
        self.outer_height_tol = outer_height_tol
        self.outer_height_ratio = outer_height_ratio
        
        # 内部维护当前高度图及有效掩码（也可以每次从点云重新生成）
        self.heightmap = np.zeros(
            (processor.grid_rows, processor.grid_cols), dtype=np.float64
        )
        self.valid_mask = np.zeros(
            (processor.grid_rows, processor.grid_cols), dtype=bool
        )
        
        # 已放置的货物记录
        self.placed_items = []
    
    def update_heightmap_from_pointcloud(self, points: np.ndarray):
        """从新的点云数据更新高度图和有效掩码。"""
        self.heightmap, self.valid_mask = self.processor.generate_heightmap_from_raw(points)

    def update_heightmap_from_ply(self, ply_path: str):
        """从 PLY 文件更新高度图和有效掩码。"""
        self.heightmap, self.valid_mask = self.processor.generate_heightmap_from_ply(ply_path)
    
    def update_heightmap_with_placement(self, 
                                         row: int, col: int,
                                         item_rows: int, item_cols: int,
                                         place_height: float, item_up: float):
        """
        在高度图上标记新放置的货物（模拟模式使用）。
        
        将货物占据的区域高度更新为 place_height + item_up。
        """
        new_height = place_height + item_up
        r_end = min(row + item_rows, self.processor.grid_rows)
        c_end = min(col + item_cols, self.processor.grid_cols)
        self.heightmap[row:r_end, col:c_end] = np.maximum(
            self.heightmap[row:r_end, col:c_end], new_height
        )
    
    def plan_placement(self, 
                       item_L: float, item_W: float, item_H: float,
                       heightmap: Optional[np.ndarray] = None,
                       xy_only: Optional[bool] = None) -> Optional[dict]:
        """
        为当前货物计算最优放置方案。
        
        Parameters
        ----------
        item_L, item_W, item_H : float
            货物原始尺寸（长、宽、高），单位米。
        heightmap : np.ndarray, optional
            当前高度图。若不提供则使用内部维护的高度图。
        xy_only : bool, optional
            覆盖实例级的 xy_only 设置。
        
        Returns
        -------
        dict or None
            最优方案，包含:
                'pose': dict          6D位姿（来自pose_generator）
                'score': float        评分
                'orientation': dict   朝向信息
                'grid_pos': (row, col)
                'place_height': float
                'item_grid_size': (rows, cols)
                'stability': dict     稳定性检测结果
            若无合法位置返回None。
        """
        if heightmap is not None:
            self.heightmap = heightmap.copy()
        
        hm = self.heightmap
        use_xy_only = xy_only if xy_only is not None else self.xy_only
        orientations = get_orientations(item_L, item_W, item_H, xy_only=use_xy_only)
        
        best_candidate = None
        best_score = -float('inf')
        
        gap_cells = max(1, int(np.ceil(PLACEMENT_GAP / self.processor.resolution)))
        
        # 粗搜索步长（格子数），用于加速。后续在最佳候选附近精搜。
        coarse_step = max(1, min(5, min(self.processor.grid_rows, 
                                         self.processor.grid_cols) // 10))
        
        for ori in orientations:
            base_x, base_y = ori['base_dims']
            up_dim = ori['up_dim']
            
            item_cols = int(np.ceil(base_x / self.processor.resolution))
            item_rows = int(np.ceil(base_y / self.processor.resolution))
            
            max_row = self.processor.grid_rows - item_rows
            max_col = self.processor.grid_cols - item_cols
            
            if max_row < 0 or max_col < 0:
                continue  # 这个朝向尺寸超出笼体
            
            # ---- 第一阶段：粗搜索 ----
            coarse_candidates = []
            for row in range(0, max_row + 1, coarse_step):
                for col in range(0, max_col + 1, coarse_step):
                    candidate = self._evaluate_position(
                        hm, row, col, item_rows, item_cols,
                        up_dim, base_x, base_y, ori
                    )
                    if candidate is not None:
                        coarse_candidates.append(candidate)
            
            # ---- 第二阶段：对Top候选精搜索 ----
            # 取粗搜索中得分最高的几个区域，在其周围做细搜索
            coarse_candidates.sort(key=lambda c: c['score'], reverse=True)
            top_n = min(5, len(coarse_candidates))
            
            refined_positions = set()
            for cand in coarse_candidates[:top_n]:
                cr, cc = cand['row'], cand['col']
                # 在 [cr-coarse_step, cr+coarse_step] 范围内精搜
                for dr in range(-coarse_step, coarse_step + 1):
                    for dc in range(-coarse_step, coarse_step + 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr <= max_row and 0 <= nc <= max_col:
                            refined_positions.add((nr, nc))
            
            # 也始终搜索边界角落（确保不遗漏角落最优解）
            for corner_row in [0, max_row]:
                for corner_col in [0, max_col]:
                    for dr in range(-coarse_step, coarse_step + 1):
                        for dc in range(-coarse_step, coarse_step + 1):
                            nr, nc = corner_row + dr, corner_col + dc
                            if 0 <= nr <= max_row and 0 <= nc <= max_col:
                                refined_positions.add((nr, nc))
            
            for (row, col) in refined_positions:
                candidate = self._evaluate_position(
                    hm, row, col, item_rows, item_cols,
                    up_dim, base_x, base_y, ori
                )
                if candidate is not None and candidate['score'] > best_score:
                    best_score = candidate['score']
                    best_candidate = candidate
        
        if best_candidate is None:
            return None
        
        # 生成6D位姿
        pose = compute_6d_pose(
            cage_origin=self.processor.cage_origin,
            row=best_candidate['row'],
            col=best_candidate['col'],
            item_grid_rows=best_candidate['item_rows'],
            item_grid_cols=best_candidate['item_cols'],
            place_height=best_candidate['place_height'],
            item_up_dim=best_candidate['up_dim'],
            orientation=best_candidate['orientation'],
            tilt_roll=best_candidate['stability']['tilt_roll'],
            tilt_pitch=best_candidate['stability']['tilt_pitch'],
            resolution=self.processor.resolution,
        )
        
        result = {
            'pose': pose,
            'score': best_candidate['score'],
            'orientation': best_candidate['orientation'],
            'grid_pos': (best_candidate['row'], best_candidate['col']),
            'place_height': best_candidate['place_height'],
            'item_grid_size': (best_candidate['item_rows'], best_candidate['item_cols']),
            'stability': best_candidate['stability'],
        }
        
        # 记录已放置货物
        self.placed_items.append({
            'dimensions': (item_L, item_W, item_H),
            'result': result,
        })
        
        return result
    
    
    def _evaluate_position(self, hm, row, col, item_rows, item_cols,
                           up_dim, base_x, base_y, ori):
        """评估单个候选位置，返回候选dict或None(不可行)。"""
        region = hm[row:row+item_rows, col:col+item_cols]
        place_height = np.max(region)
        
        # 碰撞检测
        if place_height + up_dim > self.processor.cage_height:
            return None
        
        # 稳定性检测
        stability = self.checker.check_stability(
            region, place_height, base_x, base_y
        )
        if not stability['is_stable']:
            return None
        
        # 外侧高度约束检测
        new_top = place_height + up_dim
        if not self._check_outer_height_constraint(
            hm, row, col, item_rows, item_cols, new_top
        ):
            return None
        
        # 评分
        score = self._compute_score(
            row, col, item_rows, item_cols,
            place_height, up_dim, hm
        )
        
        return {
            'row': row, 'col': col,
            'item_rows': item_rows, 'item_cols': item_cols,
            'place_height': place_height, 'up_dim': up_dim,
            'orientation': ori, 'stability': stability,
            'score': score,
        }
    
    def _check_outer_height_constraint(self, hm, row, col, 
                                        item_rows, item_cols, new_top):
        """
        外侧高度约束检查。
        
        确保放置物体后，外侧区域不会比里侧高太多。
        
        逻辑:
        - 对物体占据的每一列(X方向，col~col+item_cols)：
          1. 取该列中里侧区域(row+item_rows ~ grid_rows)的最大高度 inner_h
          2. 如果 inner_h == 0（里侧该列为空），计为违规列（不能挡住空位）
          3. 如果 inner_h > 0 且 new_top > inner_h + tolerance，计为违规列
        - 违规率 = 违规列数 / 总列数
        - 违规率 > check_ratio 时，拒绝该位置
        - 如果物体已在最里面（后面无空间），不施加约束
        
        Returns
        -------
        bool : True 表示通过约束（可以放置），False 表示违反约束
        """
        total_rows = self.processor.grid_rows
        inner_start = row + item_rows
        
        # 如果物体已在最里面（后面没有空间），无需检查
        if inner_start >= total_rows:
            return True
        
        total_cols_checked = 0
        violation_cols = 0
        
        for c in range(col, min(col + item_cols, self.processor.grid_cols)):
            total_cols_checked += 1
            
            # 该列里侧区域的最大高度
            inner_column = hm[inner_start:total_rows, c]
            inner_h = np.max(inner_column) if len(inner_column) > 0 else 0.0
            
            if inner_h <= 0:
                # 里侧该列为空 — 计为违规（不能挡住后方空位）
                violation_cols += 1
            elif new_top > inner_h + self.outer_height_tol:
                # 新物体比里侧高太多 — 违规
                violation_cols += 1
        
        if total_cols_checked == 0:
            return True
        
        violation_ratio = violation_cols / total_cols_checked
        return violation_ratio <= self.outer_height_ratio
    
    def _compute_score(self,
                       row: int, col: int,
                       item_rows: int, item_cols: int,
                       place_height: float,
                       up_dim: float,
                       heightmap: np.ndarray) -> float:
        """
        计算候选位置的综合评分。
        
        多维度评分策略：
        1. 先里后外（Y大优先）
        2. 先左后右（X小优先）
        3. 先下后上（Z小优先）
        4. 紧凑度（侧面邻接）
        5. 表面平整度（放置后顶面越平越好）
        6. 层完成度（同层高度一致加分）
        7. 空隙惩罚（底部空隙扣分）
        8. 阶梯轮廓奖励（里高外低加分）
        """
        total_rows = self.processor.grid_rows
        total_cols = self.processor.grid_cols
        cage_h = self.processor.cage_height
        
        # 货物中心行列
        center_row = row + item_rows / 2.0
        center_col = col + item_cols / 2.0
        new_top = place_height + up_dim
        
        # 1. 先里后外：row越大越好（Y大 = 里面）
        depth_score = center_row / total_rows if total_rows > 0 else 0
        
        # 2. 先左后右：col越小越好（X小 = 左边）
        left_score = 1.0 - (center_col / total_cols) if total_cols > 0 else 0
        
        # 3. 先下后上：place_height越小越好
        low_score = 1.0 - (place_height / cage_h) if cage_h > 0 else 0
        
        # 4. 紧凑度：与已有货物的贴合度
        fit_score = self._compute_fit_score(
            row, col, item_rows, item_cols, place_height, heightmap
        )
        
        # 5. 表面平整度：放置后周围区域的高度一致性
        surface_flat_score = self._compute_surface_flatness(
            row, col, item_rows, item_cols, new_top, heightmap
        )
        
        # 6. 层完成度：物体顶部与邻域高度一致时加分
        layer_score = self._compute_layer_completion(
            row, col, item_rows, item_cols, new_top, heightmap
        )
        
        # 7. 空隙惩罚：底部未填满的空间比例
        void_score = self._compute_void_penalty(
            row, col, item_rows, item_cols, place_height, heightmap
        )
        
        # 8. 阶梯轮廓奖励：里高外低的结构
        step_score = self._compute_step_profile(
            row, col, item_rows, item_cols, new_top, heightmap
        )
        
        score = (
            WEIGHT_DEPTH * depth_score +
            WEIGHT_LEFT  * left_score  +
            WEIGHT_LOW   * low_score   +
            WEIGHT_FIT   * fit_score   +
            WEIGHT_SURFACE_FLAT * surface_flat_score +
            WEIGHT_LAYER_COMPLETE * layer_score +
            WEIGHT_VOID_PENALTY * void_score +
            WEIGHT_STEP_PROFILE * step_score
        )
        
        return score
    
    def _compute_fit_score(self,
                           row: int, col: int,
                           item_rows: int, item_cols: int,
                           place_height: float,
                           heightmap: np.ndarray) -> float:
        """
        计算紧凑度评分：
        - 底部与已有货物高度越匹配越好（减少空隙）
        - 侧面紧贴已有货物或笼壁更好
        """
        score = 0.0
        total_rows, total_cols = heightmap.shape
        
        # 1. 底面匹配度：放置区域内高度的方差越小越好（说明底面平坦）
        region = heightmap[row:row+item_rows, col:col+item_cols]
        height_variance = np.var(region)
        # 方差越小分数越高,使用指数衰减
        flatness_score = np.exp(-height_variance * 100)
        score += flatness_score * 0.5
        
        # 2. 侧面邻接：四个方向上是否紧贴笼壁或已有货物
        adjacency = 0.0
        
        # 左侧
        if col == 0:
            adjacency += 1.0  # 紧贴左壁
        elif col > 0:
            left_region = heightmap[row:row+item_rows, col-1]
            if np.any(left_region >= place_height * 0.5):
                adjacency += 0.5  # 左侧有货物
        
        # 右侧
        if col + item_cols >= total_cols:
            adjacency += 1.0  # 紧贴右壁
        elif col + item_cols < total_cols:
            right_region = heightmap[row:row+item_rows, col+item_cols]
            if np.any(right_region >= place_height * 0.5):
                adjacency += 0.5
        
        # 里侧（Y正方向 = 里）
        if row + item_rows >= total_rows:
            adjacency += 1.0  # 紧贴后壁
        elif row + item_rows < total_rows:
            back_region = heightmap[row+item_rows, col:col+item_cols]
            if np.any(back_region >= place_height * 0.5):
                adjacency += 0.5
        
        # 外侧
        if row == 0:
            adjacency += 0.5  # 外壁（前面开口，给低一些分）
        elif row > 0:
            front_region = heightmap[row-1, col:col+item_cols]
            if np.any(front_region >= place_height * 0.5):
                adjacency += 0.5
                
        score += adjacency / 4.0 * 0.5
        
        return score
    
    def _compute_surface_flatness(self,
                                   row: int, col: int,
                                   item_rows: int, item_cols: int,
                                   new_top: float,
                                   heightmap: np.ndarray) -> float:
        """
        计算放置后的表面平整度评分。
        
        放置后，物体顶面高度为 new_top。检查周围邻域的高度：
        与 new_top 越接近，说明放置后表面越平，利于后续堆叠。
        """
        total_rows, total_cols = heightmap.shape
        
        # 扩展区域（物体周围一圈）的现有最高高度
        expand = 5  # 扩展几格
        r_start = max(0, row - expand)
        r_end = min(total_rows, row + item_rows + expand)
        c_start = max(0, col - expand)
        c_end = min(total_cols, col + item_cols + expand)
        
        neighbor_region = heightmap[r_start:r_end, c_start:c_end].copy()
        
        # 排除物体自身占据的区域
        # 创建掩码
        neighbor_mask = np.ones_like(neighbor_region, dtype=bool)
        rel_r = row - r_start
        rel_c = col - c_start
        neighbor_mask[rel_r:rel_r+item_rows, rel_c:rel_c+item_cols] = False
        
        neighbor_heights = neighbor_region[neighbor_mask]
        # 只考虑有高度的邻域
        neighbor_heights = neighbor_heights[neighbor_heights > 0]
        
        if len(neighbor_heights) == 0:
            return 0.5  # 没有邻居，给中等分
        
        # 高度差的标准差越小越好
        height_diff = np.abs(neighbor_heights - new_top)
        mean_diff = np.mean(height_diff)
        
        # 指数衰减：差异越小分数越高
        return np.exp(-mean_diff * 10)
    
    def _compute_layer_completion(self,
                                   row: int, col: int,
                                   item_rows: int, item_cols: int,
                                   new_top: float,
                                   heightmap: np.ndarray) -> float:
        """
        层完成度评分。
        
        如果物体顶部高度与同行（同Y方向）邻域的顶部高度一致，
        说明在完成一个"层"，加分。
        """
        total_rows, total_cols = heightmap.shape
        
        # 同一行范围内，其他列的高度
        same_row_heights = heightmap[row:row+item_rows, :]
        
        # 统计与 new_top 接近的格子占比（1cm容差内）
        tolerance = 0.02  # 2cm
        matching = np.abs(same_row_heights - new_top) < tolerance
        non_zero = same_row_heights > 0
        
        if np.sum(non_zero) == 0:
            return 0.0
        
        match_ratio = np.sum(matching & non_zero) / np.sum(non_zero)
        return match_ratio
    
    def _compute_void_penalty(self,
                               row: int, col: int,
                               item_rows: int, item_cols: int,
                               place_height: float,
                               heightmap: np.ndarray) -> float:
        """
        空隙惩罚（返回正值，会乘以负权重）。
        
        计算物体底部与实际高度图之间的空隙比例。
        空隙 = 物体底面 - 高度图表面 的体积空间。
        空隙越大说明浪费越多。
        """
        region = heightmap[row:row+item_rows, col:col+item_cols]
        
        if place_height <= 0:
            return 0.0  # 放在地面上，无空隙
        
        # 每个格子下方的空隙高度
        void_heights = place_height - region
        void_heights = np.maximum(void_heights, 0)
        
        # 空隙率 = 平均空隙高度 / 放置高度
        avg_void = np.mean(void_heights)
        void_ratio = avg_void / place_height if place_height > 0 else 0
        
        return void_ratio  # 会乘以负权重
    
    def _compute_step_profile(self,
                               row: int, col: int,
                               item_rows: int, item_cols: int,
                               new_top: float,
                               heightmap: np.ndarray) -> float:
        """
        阶梯轮廓奖励。
        
        理想结构是里高外低（从里到外阶梯下降），确保：
        1. 机械臂从外侧操作时视线不受阻挡
        2. 稳定性更好（重心靠里）
        
        若新物体顶部 <= 后方（里侧）高度，加分。
        若新物体顶部 > 前方（外侧）高度，也加分（保持阶梯）。
        """
        total_rows = heightmap.shape[0]
        score = 0.0
        
        # 检查里侧高度
        inner_start = row + item_rows
        if inner_start < total_rows:
            inner_region = heightmap[inner_start:min(inner_start+item_rows, total_rows), 
                                     col:col+item_cols]
            inner_heights = inner_region[inner_region > 0]
            if len(inner_heights) > 0:
                inner_max = np.max(inner_heights)
                if new_top <= inner_max:
                    score += 0.5  # 里侧更高或一样高，理想
                elif new_top <= inner_max + self.outer_height_tol:
                    score += 0.3  # 略高但在容差内
                else:
                    score -= 0.2  # 比里侧高太多
        
        # 检查外侧高度
        if row > 0:
            outer_region = heightmap[max(0, row-item_rows):row, 
                                     col:col+item_cols]
            outer_heights = outer_region[outer_region > 0]
            if len(outer_heights) > 0:
                outer_max = np.max(outer_heights)
                if new_top >= outer_max:
                    score += 0.5  # 比外侧高或一样高，保持阶梯
                else:
                    score += 0.2  # 比外侧低也可以
        else:
            score += 0.3  # 在最外侧，中等分
        
        return score
    
    def get_packing_stats(self) -> dict:
        """返回当前装箱统计信息。"""
        if not self.placed_items:
            return {
                'num_items': 0,
                'volume_utilization': 0.0,
                'max_height': 0.0,
            }
        
        total_volume = (self.processor.cage_width * 
                        self.processor.cage_length * 
                        self.processor.cage_height)
        
        item_volume = 0.0
        for item in self.placed_items:
            L, W, H = item['dimensions']
            item_volume += L * W * H
        
        max_height = np.max(self.heightmap)
        
        return {
            'num_items': len(self.placed_items),
            'volume_utilization': item_volume / total_volume if total_volume > 0 else 0,
            'max_height': max_height,
            'remaining_height': self.processor.cage_height - max_height,
        }
