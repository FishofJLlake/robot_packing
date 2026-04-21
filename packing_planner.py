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
    XY_ONLY_ROTATION,
    OUTER_HEIGHT_TOLERANCE, OUTER_HEIGHT_CHECK_RATIO,
    TRY_PLUS_PACKING, ENABLE_MUJOCO_SIMULATION,
    get_orientations,
)
from point_cloud_processor import PointCloudProcessor
from stability_checker import StabilityChecker
from pose_generator import compute_6d_pose, format_pose_string
from mujoco_simulator import MujocoSimulator
from scipy.spatial.transform import Rotation


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
        
        self.simulator = MujocoSimulator(
            self.processor.cage_origin, self.processor.cage_width, 
            self.processor.cage_length, self.processor.cage_height, self.processor.resolution
        )
        
        # 内部维护当前高度图及有效掩码
        self.heightmap = np.zeros(
            (processor.grid_rows, processor.grid_cols), dtype=np.float64
        )
        self.valid_mask = np.zeros(
            (processor.grid_rows, processor.grid_cols), dtype=bool
        )
        
        # 已放置的货物记录
        self.placed_items = []
        

    
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
                                         place_height: float, item_up: float, simulated_pose: Optional[dict] = None):
        """
        在高度图上标记新放置的货物（模拟模式使用）。若是模拟倾覆的位姿，则基于最大包裹尺寸更新高度图安全区域。
        """
        if simulated_pose:
            # 使用倾倒后的粗略外接盒边界（防碰撞安全）
            z_top = simulated_pose['position'][2] + max(item_rows, item_cols)*self.processor.resolution/2.0
            new_height = max(place_height + item_up, z_top)
            # 倾倒可能使物体向外占据更多网格，扩宽绘制区域以策安全
            pad = int(min(item_rows, item_cols) * 0.3)
            r_start = max(0, row - pad)
            c_start = max(0, col - pad)
            r_end = min(row + item_rows + pad, self.processor.grid_rows)
            c_end = min(col + item_cols + pad, self.processor.grid_cols)
            self.heightmap[r_start:r_end, c_start:c_end] = np.maximum(
                self.heightmap[r_start:r_end, c_start:c_end], new_height
            )
        else:
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
        
        if fill > 0.7:
            # 后期：逐步放宽外侧高度约束
            # fill=0.7 → 不变, fill=1.0 → 放宽到最大值
            progress = (fill - 0.7) / 0.3  # 0 ~ 1
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
        adaptive_tol, adaptive_ratio = self._get_adaptive_constraints()
        
        all_candidates = []
        
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
                        adaptive_tol, adaptive_ratio
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
                    adaptive_tol, adaptive_ratio
                )
                if candidate is not None:
                    coarse_candidates.append(candidate)
            
            coarse_candidates.sort(key=lambda c: c['sort_key'])
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
                    adaptive_tol, adaptive_ratio
                )
                if candidate is not None:
                    all_candidates.append(candidate)
        
        if not all_candidates:
            return None
            
        strict_cands = [c for c in all_candidates if c['stability']['stability_level'] == 'STRICT']
        plus_cands = [c for c in all_candidates if c['stability']['stability_level'] == 'PLUS']
        
        strict_cands.sort(key=lambda c: c['sort_key'])
        plus_cands.sort(key=lambda c: c['sort_key'])
        
        # 优先尝试严格稳定的策略，如果都不行，且开启了 PLUS 策略，则尝试降级方案
        if TRY_PLUS_PACKING:
            final_candidates = strict_cands + plus_cands
        else:
            final_candidates = strict_cands
            
        if not final_candidates:
            return None
            
        for cand in final_candidates:
            pose = compute_6d_pose(
                cage_origin=self.processor.cage_origin,
                row=cand['row'],
                col=cand['col'],
                item_grid_rows=cand['item_rows'],
                item_grid_cols=cand['item_cols'],
                place_height=cand['place_height'],
                item_up_dim=cand['up_dim'],
                orientation=cand['orientation'],
                tilt_roll=cand['stability']['tilt_roll'],
                tilt_pitch=cand['stability']['tilt_pitch'],
                resolution=self.processor.resolution,
            )
            
            result = {
                'pose': pose,
                'sort_key': cand['sort_key'],
                'orientation': cand['orientation'],
                'grid_pos': (cand['row'], cand['col']),
                'place_height': cand['place_height'],
                'item_grid_size': (cand['item_rows'], cand['item_cols']),
                'stability': cand['stability'],
            }
            
            if cand['stability'].get('will_tilt', False):
                if ENABLE_MUJOCO_SIMULATION:
                    is_inside, f_pos, f_quat = self.simulator.simulate_tilt(
                        hm, (item_L, item_W, item_H), pose['position'], pose['quaternion']
                    )
                    if not is_inside:
                        continue  # 物理验证失败，滑出笼子，回退到下一个候选
                    
                    rot = Rotation.from_quat(f_quat)
                    result['simulated_pose'] = {
                        'position': f_pos,
                        'quaternion': f_quat,
                        'rotation_matrix': rot.as_matrix(),
                        'orientation_euler': rot.as_euler('XYZ')
                    }
                else:
                    # 正式环境下不仿真，直接放行，但保留原始位姿作为目标
                    result['simulated_pose'] = None
            
            self.placed_items.append({
                'dimensions': (item_L, item_W, item_H),
                'result': result,
            })
            
            return result
            
        return None
    
    # ------------------------------------------------------------------
    # 位置评估
    # ------------------------------------------------------------------
    
    def _evaluate_position(self, hm, row, col, item_rows, item_cols,
                           up_dim, base_x, base_y, ori,
                           outer_tol, outer_ratio):
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
        
        # 提取用于字典序比较的物理特征
        sort_key = self._compute_lexicographical_keys(
            row, col, item_rows, item_cols,
            place_height, up_dim, hm,
            base_x=base_x, base_y=base_y
        )
        
        return {
            'row': row, 'col': col,
            'item_rows': item_rows, 'item_cols': item_cols,
            'place_height': place_height, 'up_dim': up_dim,
            'orientation': ori, 'stability': stability,
            'sort_key': sort_key,
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
    # 字典序特征提取系统 (Lexicographical Keys)
    # ------------------------------------------------------------------
    
    def _compute_max_available_area(self,
                                      hm: np.ndarray,
                                      row: int, col: int,
                                      item_rows: int, item_cols: int,
                                      place_height: float) -> float:
        """
        计算候选位置处可容纳的最大长方体底面面积（Best-Fit 策略）。

        以物品占位区为种子，沿上下左右 4 个方向逐列/逐行扩展，
        直到碰到高于 place_height 的障碍物或笼壁。

        注：使用底面面积而非体积，避免笼顶附近 available_height 缩小
        导致 fit_ratio 被人为抬高的失真问题。高度优化由 P3 (z_max) 独立负责。

        Parameters
        ----------
        hm : np.ndarray
            当前高度图。
        row, col : int
            候选放置起始行列。
        item_rows, item_cols : int
            物品占位行列数。
        place_height : float
            放置高度（物品底面高度）。

        Returns
        -------
        float : 最大可用底面面积（平方米）。
        """
        total_rows, total_cols = hm.shape

        # "同层"阈值：cell 高度不超过放置面 + 10mm 容差
        threshold = place_height + 0.01

        # --- 策略 A：先横向扩展，再纵向扩展 ---
        left_a = col
        while left_a > 0 and np.all(hm[row:row + item_rows, left_a - 1] <= threshold):
            left_a -= 1

        right_a = col + item_cols
        while right_a < total_cols and np.all(hm[row:row + item_rows, right_a] <= threshold):
            right_a += 1

        front_a = row
        while front_a > 0 and np.all(hm[front_a - 1, left_a:right_a] <= threshold):
            front_a -= 1

        back_a = row + item_rows
        while back_a < total_rows and np.all(hm[back_a, left_a:right_a] <= threshold):
            back_a += 1

        area_a = (right_a - left_a) * (back_a - front_a) * (self.processor.resolution ** 2)

        # --- 策略 B：先纵向扩展，再横向扩展 ---
        front_b = row
        while front_b > 0 and np.all(hm[front_b - 1, col:col + item_cols] <= threshold):
            front_b -= 1

        back_b = row + item_rows
        while back_b < total_rows and np.all(hm[back_b, col:col + item_cols] <= threshold):
            back_b += 1

        left_b = col
        while left_b > 0 and np.all(hm[front_b:back_b, left_b - 1] <= threshold):
            left_b -= 1

        right_b = col + item_cols
        while right_b < total_cols and np.all(hm[front_b:back_b, right_b] <= threshold):
            right_b += 1

        area_b = (right_b - left_b) * (back_b - front_b) * (self.processor.resolution ** 2)

        return max(area_a, area_b)

    def _compute_lexicographical_keys(self,
                                      row: int, col: int,
                                      item_rows: int, item_cols: int,
                                      place_height: float,
                                      up_dim: float,
                                      heightmap: np.ndarray,
                                      base_x: float = 0.0,
                                      base_y: float = 0.0) -> tuple:
        """
        计算用于字典序排序的特征元组 (用于比较，越小越好)。
        
        P0: -fit_ratio   (空间匹配度，越大越优 → 取负越小越好)
        P1: void_volume  (底部空隙体积，越小越好)
        P2: -adjacency   (四周贴合度，越大越优，取负后变为越小越好)
        P3: z_max        (放置后的最高点，越小越好)
        P4: corner_dist  (距离里左角的距离平方，越小越好)
        """
        region = heightmap[row:row+item_rows, col:col+item_cols]
        total_rows = self.processor.grid_rows
        
        # P0: fit_ratio (空间匹配度，Best-Fit 策略)
        # 使用底面面积比而非体积比，避免笼顶附近 available_height 缩小导致 ratio 失真
        item_area = base_x * base_y
        if item_area > 0:
            available_area = self._compute_max_available_area(
                heightmap, row, col, item_rows, item_cols, place_height
            )
            fit_ratio = item_area / available_area if available_area > 0 else 1.0
            fit_ratio = min(fit_ratio, 1.0)
        else:
            fit_ratio = 0.0
        
        # P1: void_volume
        if place_height <= 0:
            void_volume = 0.0
        else:
            void_heights = place_height - region
            void_heights = np.maximum(void_heights, 0)
            void_volume = np.sum(void_heights) * (self.processor.resolution ** 2)
            
        # P2: adjacency
        adjacency = self._compute_adjacency(row, col, item_rows, item_cols, place_height + up_dim, heightmap)
        
        # P3: z_max
        z_max = place_height + up_dim
        
        # P4: corner_dist
        center_row = row + item_rows / 2.0
        center_col = col + item_cols / 2.0
        dist_y = total_rows - center_row   # 距离里侧的行数
        dist_x = center_col                # 距离左侧的列数
        corner_dist = dist_y**2 + dist_x**2
        
        # 防止浮点比较异常，圆整 (fit_ratio 保留2位小数做适度分桶)
        return (-round(float(fit_ratio), 2), round(float(void_volume), 6), -round(float(adjacency), 4), round(float(z_max), 4), round(float(corner_dist), 2))
        
    def _compute_adjacency(self, row, col, item_rows, item_cols, new_top, heightmap):
        """就算物体四周紧贴现有箱体或笼壁的比例，总分为4。"""
        adjacency = 0.0
        total_rows, total_cols = heightmap.shape
        tol = 0.05 # 5cm的高差内视为有效贴靠
        
        # 左侧
        if col == 0:
            adjacency += 1.0
        else:
            left_region = heightmap[row:row+item_rows, col-1]
            adjacency += np.sum(left_region >= new_top - tol) / item_rows
            
        # 右侧
        if col + item_cols >= total_cols:
            adjacency += 1.0
        else:
            right_region = heightmap[row:row+item_rows, col+item_cols]
            adjacency += np.sum(right_region >= new_top - tol) / item_rows
            
        # 里侧
        if row + item_rows >= total_rows:
            adjacency += 1.0
        else:
            back_region = heightmap[row+item_rows, col:col+item_cols]
            adjacency += np.sum(back_region >= new_top - tol) / item_cols
            
        # 外侧/前部 (门是打开的，没有物理壁可以依靠，不计算贴合分)
        if row == 0:
            pass  # 不提供贴合度分数

        else:
            front_region = heightmap[row-1, col:col+item_cols]
            adjacency += np.sum(front_region >= new_top - tol) / item_cols
            
        return adjacency

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    
    def get_packing_stats(self) -> dict:
        """返回当前装箱统计信息。"""
        max_height = np.max(self.heightmap) if self.heightmap is not None else 0.0
        
        if not self.placed_items:
            return {
                'num_items': 0,
                'volume_utilization': 0.0,
                'max_height': max_height,
                'remaining_height': self.processor.cage_height - max_height,
                'fill_ratio': 0.0,
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
