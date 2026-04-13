"""
3D装箱系统 — 核心装箱规划器

实现基于高度图的在线装箱算法，支持多种朝向，
评分规则为：先里后外、先左后右、先下后上。

增强功能:
  1. XY-only旋转开关
  2. 外侧高度约束（比例检查）
  3. 多维度评分策略（平整度、层完成度、空隙惩罚、阶梯奖励）
  4. 自适应权重策略（根据填充状态动态调整）
  5. 天际线候选生成（更智能的位置搜索）
  6. 动态约束松弛（仅外侧高度约束，支撑面积保持严格）
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
    
    自适应策略:
    - 根据当前填充状态（高度占比）动态调整评分权重
    - 初期强调结构建设，后期强调空间利用
    - 外侧高度约束在笼子接近满载时适度放宽
    - 支撑面积比始终保持严格（安全约束不松弛）
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
        self.outer_height_tol_base = outer_height_tol
        self.outer_height_ratio_base = outer_height_ratio
        
        # 内部维护当前高度图及有效掩码
        self.heightmap = np.zeros(
            (processor.grid_rows, processor.grid_cols), dtype=np.float64
        )
        self.valid_mask = np.zeros(
            (processor.grid_rows, processor.grid_cols), dtype=bool
        )
        
        # 已放置的货物记录
        self.placed_items = []
        
        # 基础权重（从config读取）
        self._base_weights = {
            'depth': WEIGHT_DEPTH,
            'left': WEIGHT_LEFT,
            'low': WEIGHT_LOW,
            'fit': WEIGHT_FIT,
            'surface_flat': WEIGHT_SURFACE_FLAT,
            'layer_complete': WEIGHT_LAYER_COMPLETE,
            'void_penalty': WEIGHT_VOID_PENALTY,
            'step_profile': WEIGHT_STEP_PROFILE,
        }
    
    # ------------------------------------------------------------------
    # 高度图更新
    # ------------------------------------------------------------------
    
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
        """
        new_height = place_height + item_up
        r_end = min(row + item_rows, self.processor.grid_rows)
        c_end = min(col + item_cols, self.processor.grid_cols)
        self.heightmap[row:r_end, col:c_end] = np.maximum(
            self.heightmap[row:r_end, col:c_end], new_height
        )
    
    # ------------------------------------------------------------------
    # 填充状态评估
    # ------------------------------------------------------------------
    
    def _get_fill_ratio(self) -> float:
        """
        计算当前笼体填充率（基于高度占比）。
        
        使用最大高度 / 笼高作为主要指标，
        同时考虑高度图中非零区域的面积占比。
        
        Returns
        -------
        float : [0, 1] 填充率
        """
        max_h = np.max(self.heightmap) if self.heightmap.size > 0 else 0
        height_ratio = max_h / self.processor.cage_height if self.processor.cage_height > 0 else 0
        
        # 面积覆盖率
        area_ratio = np.mean(self.heightmap > 0)
        
        # 综合指标（高度占比为主，面积为辅）
        fill_ratio = 0.6 * height_ratio + 0.4 * area_ratio
        return np.clip(fill_ratio, 0, 1)
    
    def _get_adaptive_weights(self) -> dict:
        """
        根据当前填充状态动态调整评分权重。
        
        策略:
        - 初期 (fill < 0.3): 强调 depth/low，建立良好底层结构
        - 中期 (0.3~0.7):  平衡所有指标
        - 后期 (fill > 0.7): 增大 fit/void_penalty/surface_flat，
                              减小 depth（不再执着先里后外，空哪放哪）
        """
        fill = self._get_fill_ratio()
        w = self._base_weights.copy()
        
        if fill < 0.3:
            # 初期：稳定底层结构
            # depth 和 low 保持高权重，其他维持基础值
            pass
        elif fill < 0.7:
            # 中期：开始更注重紧凑
            w['fit'] *= 1.5
            w['void_penalty'] *= 1.3
            w['surface_flat'] *= 1.2
        else:
            # 后期：最大化空间利用
            w['depth'] *= 0.5        # 减弱"先里后外"——空间不够时不再执着
            w['left'] *= 0.5         # 减弱"先左后右"
            w['fit'] *= 2.0          # 大幅增强紧凑度
            w['void_penalty'] *= 2.0 # 大幅增强空隙惩罚
            w['surface_flat'] *= 1.5 # 增强平整度
            w['layer_complete'] *= 1.5
        
        return w
    
    def _get_adaptive_constraints(self) -> Tuple[float, float]:
        """
        动态约束松弛：仅对外侧高度约束进行松弛。
        
        注意：支撑面积比始终保持严格60%，不允许松弛！
        （外侧无支撑会导致倾倒出笼，存在安全风险）
        
        Returns
        -------
        (outer_height_tol, outer_height_ratio) : 动态调整后的值
        """
        fill = self._get_fill_ratio()
        
        tol = self.outer_height_tol_base
        ratio = self.outer_height_ratio_base
        
        if fill > 0.5:
            # 后期：逐步放宽外侧高度约束
            # fill=0.5 → 不变, fill=1.0 → 放宽到最大值
            progress = (fill - 0.5) / 0.5  # 0 ~ 1
            tol = self.outer_height_tol_base + progress * 0.03   # 5cm → 8cm
            ratio = self.outer_height_ratio_base + progress * 0.13  # 0.67 → 0.80
            ratio = min(ratio, 0.85)  # 上限
        
        return tol, ratio
    
    # ------------------------------------------------------------------
    # 天际线候选生成
    # ------------------------------------------------------------------
    
    def _extract_skyline_candidates(self, hm: np.ndarray,
                                      item_rows: int, item_cols: int,
                                      max_row: int, max_col: int) -> set:
        """
        基于天际线（高度轮廓转折点）生成候选位置。
        
        天际线 = 高度图在X/Y方向上的高度突变点位。
        在这些位置放置物品最容易贴合已有物品边缘。
        
        Returns
        -------
        set of (row, col) : 候选位置集合
        """
        candidates = set()
        
        # 1. X方向天际线：每行扫描，找高度变化点
        for r in range(0, max_row + 1, max(1, item_rows // 2)):
            row_heights = hm[r, :]
            # 找高度变化的列位置
            diff = np.diff(row_heights)
            change_cols = np.where(np.abs(diff) > 0.005)[0]  # 5mm阈值
            for c in change_cols:
                # 在变化点前后都试放
                for dc in [-item_cols, 0, 1]:
                    nc = c + dc
                    if 0 <= nc <= max_col:
                        candidates.add((r, nc))
        
        # 2. Y方向天际线：每列扫描，找高度变化点
        for c in range(0, max_col + 1, max(1, item_cols // 2)):
            col_heights = hm[:, c]
            diff = np.diff(col_heights)
            change_rows = np.where(np.abs(diff) > 0.005)[0]
            for r in change_rows:
                for dr in [-item_rows, 0, 1]:
                    nr = r + dr
                    if 0 <= nr <= max_row:
                        candidates.add((nr, c))
        
        # 3. 已放置物品的边缘紧贴位置
        for item in self.placed_items:
            result = item['result']
            pr, pc = result['grid_pos']
            ir, ic = result['item_grid_size']
            
            # 物品四个边缘外侧位置
            edge_positions = [
                (pr - item_rows, pc),           # 外侧（前方）
                (pr + ir, pc),                  # 里侧（后方）
                (pr, pc - item_cols),           # 左侧
                (pr, pc + ic),                  # 右侧
                (pr - item_rows, pc + ic),      # 外右角
                (pr + ir, pc - item_cols),      # 里左角
            ]
            for (nr, nc) in edge_positions:
                if 0 <= nr <= max_row and 0 <= nc <= max_col:
                    candidates.add((nr, nc))
        
        # 4. "平坦区域"起始点：找高度一致区域的边界
        # 逐行扫描找到从0→非0的转折
        for r in range(0, self.processor.grid_rows, max(1, item_rows)):
            nonzero_cols = np.where(hm[r, :] > 0)[0]
            if len(nonzero_cols) > 0:
                # 非零区域的左右边界
                first_c = nonzero_cols[0]
                last_c = nonzero_cols[-1]
                for c in [max(0, first_c - item_cols), first_c, 
                           last_c + 1, max(0, last_c - item_cols + 1)]:
                    if 0 <= c <= max_col and 0 <= r <= max_row:
                        candidates.add((r, c))
        
        return candidates
    
    # ------------------------------------------------------------------
    # 核心：放置规划
    # ------------------------------------------------------------------
    
    def plan_placement(self, 
                       item_L: float, item_W: float, item_H: float,
                       heightmap: Optional[np.ndarray] = None,
                       xy_only: Optional[bool] = None) -> Optional[dict]:
        """
        为当前货物计算最优放置方案。
        
        搜索策略：粗搜索 + 天际线候选 + 精搜索三阶段。
        
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
        """
        if heightmap is not None:
            self.heightmap = heightmap.copy()
        
        hm = self.heightmap
        use_xy_only = xy_only if xy_only is not None else self.xy_only
        orientations = get_orientations(item_L, item_W, item_H, xy_only=use_xy_only)
        
        # 获取自适应权重和约束
        adaptive_weights = self._get_adaptive_weights()
        adaptive_tol, adaptive_ratio = self._get_adaptive_constraints()
        
        best_candidate = None
        best_score = -float('inf')
        
        # 粗搜索步长
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
                continue
            
            # ---- 阶段1：粗搜索 ----
            coarse_candidates = []
            for row in range(0, max_row + 1, coarse_step):
                for col in range(0, max_col + 1, coarse_step):
                    candidate = self._evaluate_position(
                        hm, row, col, item_rows, item_cols,
                        up_dim, base_x, base_y, ori,
                        adaptive_weights, adaptive_tol, adaptive_ratio
                    )
                    if candidate is not None:
                        coarse_candidates.append(candidate)
            
            # ---- 阶段2：天际线候选 ----
            skyline_positions = self._extract_skyline_candidates(
                hm, item_rows, item_cols, max_row, max_col
            )
            for (row, col) in skyline_positions:
                candidate = self._evaluate_position(
                    hm, row, col, item_rows, item_cols,
                    up_dim, base_x, base_y, ori,
                    adaptive_weights, adaptive_tol, adaptive_ratio
                )
                if candidate is not None:
                    coarse_candidates.append(candidate)
            
            # ---- 阶段3：精搜索（Top-N 附近 + 角落）----
            coarse_candidates.sort(key=lambda c: c['score'], reverse=True)
            top_n = min(8, len(coarse_candidates))  # 增加精搜区域
            
            refined_positions = set()
            for cand in coarse_candidates[:top_n]:
                cr, cc = cand['row'], cand['col']
                for dr in range(-coarse_step, coarse_step + 1):
                    for dc in range(-coarse_step, coarse_step + 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr <= max_row and 0 <= nc <= max_col:
                            refined_positions.add((nr, nc))
            
            # 角落搜索
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
                    up_dim, base_x, base_y, ori,
                    adaptive_weights, adaptive_tol, adaptive_ratio
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
    
    # ------------------------------------------------------------------
    # 位置评估
    # ------------------------------------------------------------------
    
    def _evaluate_position(self, hm, row, col, item_rows, item_cols,
                           up_dim, base_x, base_y, ori,
                           weights, outer_tol, outer_ratio):
        """评估单个候选位置，返回候选dict或None(不可行)。"""
        region = hm[row:row+item_rows, col:col+item_cols]
        place_height = np.max(region)
        
        # 碰撞检测
        if place_height + up_dim > self.processor.cage_height:
            return None
        
        # 稳定性检测（始终保持严格，不松弛！）
        stability = self.checker.check_stability(
            region, place_height, base_x, base_y
        )
        if not stability['is_stable']:
            return None
        
        # 外侧高度约束检测（使用自适应阈值）
        new_top = place_height + up_dim
        if not self._check_outer_height_constraint(
            hm, row, col, item_rows, item_cols, new_top,
            outer_tol, outer_ratio
        ):
            return None
        
        # 评分（使用自适应权重）
        score = self._compute_score(
            row, col, item_rows, item_cols,
            place_height, up_dim, hm, weights
        )
        
        return {
            'row': row, 'col': col,
            'item_rows': item_rows, 'item_cols': item_cols,
            'place_height': place_height, 'up_dim': up_dim,
            'orientation': ori, 'stability': stability,
            'score': score,
        }
    
    # ------------------------------------------------------------------
    # 外侧高度约束
    # ------------------------------------------------------------------
    
    def _check_outer_height_constraint(self, hm, row, col, 
                                        item_rows, item_cols, new_top,
                                        outer_tol=None, outer_ratio=None):
        """
        外侧高度约束检查。
        
        逻辑:
        - 对物体占据的每一列(X方向，col~col+item_cols)：
          1. 取该列中里侧区域(row+item_rows ~ grid_rows)的最大高度 inner_h
          2. 如果 inner_h == 0（里侧该列为空），计为违规列（不能挡住空位）
          3. 如果 inner_h > 0 且 new_top > inner_h + tolerance，计为违规列
        - 违规率 = 违规列数 / 总列数
        - 违规率 > check_ratio 时，拒绝该位置
        """
        tol = outer_tol if outer_tol is not None else self.outer_height_tol_base
        ratio = outer_ratio if outer_ratio is not None else self.outer_height_ratio_base
        
        total_rows = self.processor.grid_rows
        inner_start = row + item_rows
        
        if inner_start >= total_rows:
            return True
        
        total_cols_checked = 0
        violation_cols = 0
        
        for c in range(col, min(col + item_cols, self.processor.grid_cols)):
            total_cols_checked += 1
            inner_column = hm[inner_start:total_rows, c]
            inner_h = np.max(inner_column) if len(inner_column) > 0 else 0.0
            
            if inner_h <= 0:
                violation_cols += 1
            elif new_top > inner_h + tol:
                violation_cols += 1
        
        if total_cols_checked == 0:
            return True
        
        violation_ratio = violation_cols / total_cols_checked
        return violation_ratio <= ratio
    
    # ------------------------------------------------------------------
    # 评分系统
    # ------------------------------------------------------------------
    
    def _compute_score(self,
                       row: int, col: int,
                       item_rows: int, item_cols: int,
                       place_height: float,
                       up_dim: float,
                       heightmap: np.ndarray,
                       weights: Optional[dict] = None) -> float:
        """
        计算候选位置的综合评分（使用自适应权重）。
        """
        w = weights or self._base_weights
        
        total_rows = self.processor.grid_rows
        total_cols = self.processor.grid_cols
        cage_h = self.processor.cage_height
        
        center_row = row + item_rows / 2.0
        center_col = col + item_cols / 2.0
        new_top = place_height + up_dim
        
        # 1. 先里后外
        depth_score = center_row / total_rows if total_rows > 0 else 0
        
        # 2. 先左后右
        left_score = 1.0 - (center_col / total_cols) if total_cols > 0 else 0
        
        # 3. 先下后上
        low_score = 1.0 - (place_height / cage_h) if cage_h > 0 else 0
        
        # 4. 紧凑度
        fit_score = self._compute_fit_score(
            row, col, item_rows, item_cols, place_height, heightmap
        )
        
        # 5. 表面平整度
        surface_flat_score = self._compute_surface_flatness(
            row, col, item_rows, item_cols, new_top, heightmap
        )
        
        # 6. 层完成度
        layer_score = self._compute_layer_completion(
            row, col, item_rows, item_cols, new_top, heightmap
        )
        
        # 7. 空隙惩罚
        void_score = self._compute_void_penalty(
            row, col, item_rows, item_cols, place_height, heightmap
        )
        
        # 8. 阶梯轮廓奖励
        step_score = self._compute_step_profile(
            row, col, item_rows, item_cols, new_top, heightmap
        )
        
        score = (
            w['depth'] * depth_score +
            w['left']  * left_score  +
            w['low']   * low_score   +
            w['fit']   * fit_score   +
            w['surface_flat'] * surface_flat_score +
            w['layer_complete'] * layer_score +
            w['void_penalty'] * void_score +
            w['step_profile'] * step_score
        )
        
        return score
    
    # ------------------------------------------------------------------
    # 评分子项
    # ------------------------------------------------------------------
    
    def _compute_fit_score(self,
                           row: int, col: int,
                           item_rows: int, item_cols: int,
                           place_height: float,
                           heightmap: np.ndarray) -> float:
        """
        紧凑度评分：底面匹配度 + 侧面邻接度。
        """
        score = 0.0
        total_rows, total_cols = heightmap.shape
        
        # 1. 底面匹配度
        region = heightmap[row:row+item_rows, col:col+item_cols]
        height_variance = np.var(region)
        flatness_score = np.exp(-height_variance * 100)
        score += flatness_score * 0.5
        
        # 2. 侧面邻接
        adjacency = 0.0
        
        if col == 0:
            adjacency += 1.0
        elif col > 0:
            left_region = heightmap[row:row+item_rows, col-1]
            if np.any(left_region >= place_height * 0.5):
                adjacency += 0.5
        
        if col + item_cols >= total_cols:
            adjacency += 1.0
        elif col + item_cols < total_cols:
            right_region = heightmap[row:row+item_rows, col+item_cols]
            if np.any(right_region >= place_height * 0.5):
                adjacency += 0.5
        
        if row + item_rows >= total_rows:
            adjacency += 1.0
        elif row + item_rows < total_rows:
            back_region = heightmap[row+item_rows, col:col+item_cols]
            if np.any(back_region >= place_height * 0.5):
                adjacency += 0.5
        
        if row == 0:
            adjacency += 0.5
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
        """放置后表面平整度评分。"""
        total_rows, total_cols = heightmap.shape
        
        expand = 5
        r_start = max(0, row - expand)
        r_end = min(total_rows, row + item_rows + expand)
        c_start = max(0, col - expand)
        c_end = min(total_cols, col + item_cols + expand)
        
        neighbor_region = heightmap[r_start:r_end, c_start:c_end].copy()
        
        neighbor_mask = np.ones_like(neighbor_region, dtype=bool)
        rel_r = row - r_start
        rel_c = col - c_start
        neighbor_mask[rel_r:rel_r+item_rows, rel_c:rel_c+item_cols] = False
        
        neighbor_heights = neighbor_region[neighbor_mask]
        neighbor_heights = neighbor_heights[neighbor_heights > 0]
        
        if len(neighbor_heights) == 0:
            return 0.5
        
        height_diff = np.abs(neighbor_heights - new_top)
        mean_diff = np.mean(height_diff)
        return np.exp(-mean_diff * 10)
    
    def _compute_layer_completion(self,
                                   row: int, col: int,
                                   item_rows: int, item_cols: int,
                                   new_top: float,
                                   heightmap: np.ndarray) -> float:
        """层完成度评分。"""
        same_row_heights = heightmap[row:row+item_rows, :]
        
        tolerance = 0.02
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
        """空隙惩罚（正值，乘以负权重）。"""
        region = heightmap[row:row+item_rows, col:col+item_cols]
        
        if place_height <= 0:
            return 0.0
        
        void_heights = place_height - region
        void_heights = np.maximum(void_heights, 0)
        
        avg_void = np.mean(void_heights)
        void_ratio = avg_void / place_height if place_height > 0 else 0
        return void_ratio
    
    def _compute_step_profile(self,
                               row: int, col: int,
                               item_rows: int, item_cols: int,
                               new_top: float,
                               heightmap: np.ndarray) -> float:
        """阶梯轮廓奖励（里高外低）。"""
        total_rows = heightmap.shape[0]
        score = 0.0
        
        inner_start = row + item_rows
        if inner_start < total_rows:
            inner_region = heightmap[inner_start:min(inner_start+item_rows, total_rows), 
                                     col:col+item_cols]
            inner_heights = inner_region[inner_region > 0]
            if len(inner_heights) > 0:
                inner_max = np.max(inner_heights)
                if new_top <= inner_max:
                    score += 0.5
                elif new_top <= inner_max + self.outer_height_tol_base:
                    score += 0.3
                else:
                    score -= 0.2
        
        if row > 0:
            outer_region = heightmap[max(0, row-item_rows):row, 
                                     col:col+item_cols]
            outer_heights = outer_region[outer_region > 0]
            if len(outer_heights) > 0:
                outer_max = np.max(outer_heights)
                if new_top >= outer_max:
                    score += 0.5
                else:
                    score += 0.2
        else:
            score += 0.3
        
        return score
    
    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    
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
        fill_ratio = self._get_fill_ratio()
        
        return {
            'num_items': len(self.placed_items),
            'volume_utilization': item_volume / total_volume if total_volume > 0 else 0,
            'max_height': max_height,
            'remaining_height': self.processor.cage_height - max_height,
            'fill_ratio': fill_ratio,
        }
