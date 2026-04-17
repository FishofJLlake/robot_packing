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
# PLY 文件路径与点云处理参数
# ============================================================
PLY_FILE_PATH = None
# 真实点云裁剪范围 (单位与点云文件自身一致)
# 格式为 (min_val, max_val)。如果没有限制则用 None
PLY_CROP_X = (-0.92, -0.02)
PLY_CROP_Y = (0.0, 0.60)
PLY_CROP_Z = (-1.2, 0.50)

# ============================================================
# XY-only 旋转开关
# True 时只考虑绕Z轴旋转（保持底面朝下），即2种朝向
# False 时考虑全部6种朝向
# ============================================================
XY_ONLY_ROTATION = False

# ============================================================
# 外侧高度约束
# ============================================================
# 同一列（X方向相同位置），外侧物体顶部不能比里侧物体顶部高超过此值
OUTER_HEIGHT_TOLERANCE = 0.10       # 10cm
# 违规列占有效列的比例阈值：超过此比例则拒绝放置
OUTER_HEIGHT_CHECK_RATIO = 0.67     # 2/3

# ============================================================
# 稳定性阈值
# ============================================================
MIN_SUPPORT_RATIO = 0.60      # 底部至少 60% 面积有支撑
MAX_TILT_ANGLE    = 5.0       # 最大允许倾斜角度（度）
SUPPORT_HEIGHT_TOLERANCE = 0.05  # 支撑高度容差 超参数：平面阈值，比如 5cm 内起伏皆视为支撑平面

# ============================================================
# 评分系统
# ============================================================
# （已弃用）改为使用无调参的字典序评估 (Lexicographical evaluation)

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

def get_orientations(L, W, H, xy_only=False):
    """
    给定物体原始尺寸 (L, W, H)，返回可用朝向下的参数。
    
    Parameters
    ----------
    L, W, H : float
        物体尺寸。
    xy_only : bool
        若为True，只返回物体底面朝下的朝向（仅绕Z轴旋转0°/90°），
        即只考虑在物体坐标系xy方向旋转。
    
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
    if xy_only:
        # 仅考虑底面朝下（Z轴不变），绕Z轴旋转0°/90°
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
        ]
    else:
        # 全部6种朝向
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
