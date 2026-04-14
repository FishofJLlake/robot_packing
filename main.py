"""
3D装箱系统 — 主入口与演示

提供多种模式:
1. 模拟演示模式: 使用合成数据模拟多个货物逐个到来的装箱过程
2. PLY 文件模式: 从真实点云加载初始笼内状态，在此基础上放置新物品
3. 实时模式接口: 可集成到实际系统中

命令行参数:
  --ply <path>   从PLY文件加载初始笼内状态
  --xy-only      仅XY旋转模式（物体底面朝下，绕Z轴旋转0°/90°）
  --uneven       不平底面装箱演示
"""
import numpy as np
import sys
import argparse
from typing import List, Tuple

from config import (
    CAGE_WIDTH, CAGE_LENGTH, CAGE_HEIGHT,
    HEIGHTMAP_RESOLUTION, PLACEMENT_GAP,
    XY_ONLY_ROTATION,
)
from point_cloud_processor import PointCloudProcessor
from stability_checker import StabilityChecker
from packing_planner import PackingPlanner
from pose_generator import format_pose_string
from visualizer import (
    visualize_heightmap,
    visualize_packing_2d,
    visualize_packing_3d,
)


def create_system(cage_origin=(0.0, 0.0, 0.0),
                  cage_width=CAGE_WIDTH,
                  cage_length=CAGE_LENGTH,
                  cage_height=CAGE_HEIGHT,
                  resolution=HEIGHTMAP_RESOLUTION,
                  xy_only=XY_ONLY_ROTATION) -> PackingPlanner:
    """
    创建装箱系统实例。
    
    Parameters
    ----------
    cage_origin : tuple
        笼体原点 (x, y, z)。
    cage_width, cage_length, cage_height : float
        笼体尺寸。
    resolution : float
        高度图分辨率。
    xy_only : bool
        仅XY旋转模式。
    
    Returns
    -------
    PackingPlanner
        装箱规划器。
    """
    processor = PointCloudProcessor(
        cage_origin=cage_origin,
        cage_width=cage_width,
        cage_length=cage_length,
        cage_height=cage_height,
        resolution=resolution,
    )
    
    checker = StabilityChecker(resolution=resolution)
    planner = PackingPlanner(processor, checker, xy_only=xy_only)
    
    return planner


def simulate_packing(planner: PackingPlanner,
                     items: List[Tuple[float, float, float]],
                     verbose: bool = True) -> List[dict]:
    """
    模拟在线装箱过程：逐个放置货物。
    
    Parameters
    ----------
    planner : PackingPlanner
        装箱规划器。
    items : list of (L, W, H)
        按到达顺序排列的货物尺寸列表。
    verbose : bool
        是否打印详细信息。
    
    Returns
    -------
    list of dict
        成功放置的货物结果列表。
    """
    results = []
    
    if verbose:
        print("=" * 60)
        print("  3D视觉装箱系统 — 模拟演示")
        print("=" * 60)
        print(f"  笼体尺寸: {planner.processor.cage_width:.2f} × "
              f"{planner.processor.cage_length:.2f} × "
              f"{planner.processor.cage_height:.2f} m")
        print(f"  高度图分辨率: {planner.processor.resolution*100:.0f} cm")
        print(f"  待放置货物数: {len(items)}")
        print(f"  货物间隙: {PLACEMENT_GAP*100:.0f} cm")
        print(f"  仅XY旋转: {'是' if planner.xy_only else '否'}")
        print("=" * 60)
    
    for i, (L, W, H) in enumerate(items):
        if verbose:
            print(f"\n--- 货物 #{i+1}: {L:.3f} × {W:.3f} × {H:.3f} m ---")
        
        result = planner.plan_placement(L, W, H)
        
        if result is None:
            if verbose:
                print(f"  ❌ 无法放置！笼内剩余空间不足或无稳定位置。")
            continue
        
        # 更新高度图（模拟模式）
        row, col = result['grid_pos']
        item_rows, item_cols = result['item_grid_size']
        planner.update_heightmap_with_placement(
            row, col, item_rows, item_cols,
            result['place_height'], result['orientation']['up_dim']
        )
        
        results.append(result)
        
        if verbose:
            print(f"  ✅ 放置成功！")
            print(f"  朝向: {result['orientation']['desc']}")
            print(f"  网格位置: row={row}, col={col}")
            print(f"  放置高度: {result['place_height']:.4f} m")
            print(f"  排序特征 (空隙, -贴靠, 高度, 角落距离): {result['sort_key']}")
            print(f"  支撑面积: {result['stability']['support_ratio']:.1%}")
            print(f"  底面倾斜: {result['stability']['tilt_angle']:.2f}°")
            print(f"  {format_pose_string(result['pose'])}")
    
    if verbose:
        stats = planner.get_packing_stats()
        print("\n" + "=" * 60)
        print("  装箱统计")
        print("=" * 60)
        print(f"  成功放置: {stats['num_items']} / {len(items)} 个")
        print(f"  体积利用率: {stats['volume_utilization']:.1%}")
        print(f"  最大高度: {stats['max_height']:.3f} m")
        print(f"  剩余高度: {stats['remaining_height']:.3f} m")
        print("=" * 60)
    
    return results


def run_demo(xy_only=False):
    """运行演示：模拟多种尺寸快递箱的装箱过程。"""
    
    # 创建系统
    planner = create_system(xy_only=xy_only)
    
    # 模拟来料：一批不同尺寸的快递箱
    # 尺寸单位：米 (L, W, H)
    items = [
        (0.40, 0.30, 0.25),   # 中等快递箱
        (0.35, 0.25, 0.20),   # 较小快递箱
        (0.50, 0.35, 0.30),   # 大号快递箱
        (0.30, 0.25, 0.15),   # 小号快递箱
        (0.45, 0.30, 0.25),   # 中等快递箱
        (0.25, 0.20, 0.15),   # 迷你快递箱
        (0.40, 0.35, 0.30),   # 中大号快递箱
        (0.35, 0.30, 0.20),   # 中等快递箱
        (0.30, 0.20, 0.15),   # 小号
        (0.45, 0.35, 0.25),   # 中大号
        (0.25, 0.25, 0.20),   # 正方形小箱
        (0.50, 0.40, 0.30),   # 超大号
    ]
    
    # 执行模拟装箱
    results = simulate_packing(planner, items, verbose=True)
    
    # 可视化
    print("\n正在生成可视化...")
    
    # 2D 高度图
    visualize_heightmap(planner.heightmap, title="装箱完成后的高度图")
    
    # 2D 俯视图（叠加货物轮廓）
    visualize_packing_2d(
        planner.heightmap,
        planner.placed_items,
        title="装箱俯视图"
    )
    
    # 3D 可视化
    visualize_packing_3d(
        cage_origin=planner.processor.cage_origin,
        cage_width=planner.processor.cage_width,
        cage_length=planner.processor.cage_length,
        cage_height=planner.processor.cage_height,
        placed_items=planner.placed_items,
        title="3D装箱结果"
    )


def run_demo_ply(ply_path: str, xy_only: bool = False):
    """
    PLY 文件模式演示：从真实点云加载初始笼内状态，在此基础上放置新物品。
    
    Parameters
    ----------
    ply_path : str
        PLY 文件路径。
    xy_only : bool
        仅XY旋转模式。
    """
    print("\n" + "=" * 60)
    print("  PLY 文件装箱演示")
    print("=" * 60)
    
    # 创建系统
    planner = create_system(xy_only=xy_only)
    
    # 从 PLY 文件加载初始高度图
    print("\n正在加载 PLY 文件...")
    planner.update_heightmap_from_ply(ply_path)
    
    # 显示初始高度图
    print("\n初始高度图（来自 PLY 文件）：")
    print(f"  最大高度: {np.max(planner.heightmap):.3f} m")
    print(f"  有效覆盖率: {np.mean(planner.valid_mask):.1%}")
    
    visualize_heightmap(planner.heightmap, title="PLY 初始高度图")
    
    # 在已有高度图基础上放置新物品
    items = [
        (0.40, 0.30, 0.25),   # 中等快递箱
        (0.35, 0.25, 0.20),   # 较小快递箱
        (0.30, 0.25, 0.15),   # 小号快递箱
        (0.25, 0.20, 0.15),   # 迷你快递箱
    ]
    
    print(f"\n在PLY初始状态上放置 {len(items)} 个新物品...")
    results = simulate_packing(planner, items, verbose=True)
    
    # 可视化
    print("\n正在生成可视化...")
    
    visualize_heightmap(planner.heightmap, title="PLY + 新物品 高度图")
    
    visualize_packing_2d(
        planner.heightmap,
        planner.placed_items,
        title="PLY + 新物品 俯视图"
    )
    
    visualize_packing_3d(
        cage_origin=planner.processor.cage_origin,
        cage_width=planner.processor.cage_width,
        cage_length=planner.processor.cage_length,
        cage_height=planner.processor.cage_height,
        placed_items=planner.placed_items,
        title="PLY + 新物品 3D视图"
    )


def run_demo_uneven_surface(xy_only=False):
    """
    演示：不平底面的装箱场景。
    模拟已有几个大小不一的箱子在底层，上面再放新箱子。
    """
    print("\n" + "=" * 60)
    print("  不平底面装箱演示")
    print("=" * 60)
    
    planner = create_system(xy_only=xy_only)
    
    # 先放置几个底层箱子，制造不平底面
    bottom_items = [
        (0.40, 0.50, 0.30),   # 底层-1
        (0.35, 0.45, 0.25),   # 底层-2（比1矮5cm）
    ]
    
    print("\n-- 阶段1: 放置底层货物 --")
    simulate_packing(planner, bottom_items, verbose=True)
    
    # 再放置上层货物（需要处理不平底面）
    top_items = [
        (0.30, 0.25, 0.20),   # 顶层-1
        (0.35, 0.30, 0.15),   # 顶层-2
        (0.25, 0.20, 0.15),   # 顶层-3
    ]
    
    print("\n-- 阶段2: 在不平底面上放置新货物 --")
    simulate_packing(planner, top_items, verbose=True)
    
    # 可视化
    visualize_heightmap(planner.heightmap, title="不平底面装箱 — 高度图")
    
    visualize_packing_3d(
        cage_origin=planner.processor.cage_origin,
        cage_width=planner.processor.cage_width,
        cage_length=planner.processor.cage_length,
        cage_height=planner.processor.cage_height,
        placed_items=planner.placed_items,
        title="不平底面装箱 — 3D视图"
    )


def realtime_interface():
    """
    实时模式接口示例。
    
    展示如何集成到实际系统中：
    每次收到新的点云和货物尺寸时调用。
    """
    planner = create_system()
    
    print("实时装箱接口已启动。")
    print("每次调用 process_new_item(planner, points, L, W, H) 进行决策。")
    
    return planner


def process_new_item(planner: PackingPlanner,
                     point_cloud: np.ndarray,
                     item_L: float, item_W: float, item_H: float) -> dict:
    """
    处理一个新到来的货物（实时模式）。
    
    Parameters
    ----------
    planner : PackingPlanner
        装箱规划器。
    point_cloud : np.ndarray, shape (N, 3)
        当前笼体的实时3D点云。
    item_L, item_W, item_H : float
        当前货物尺寸（米）。
    
    Returns
    -------
    dict or None
        放置结果。包含 'pose' 等信息。
    """
    # 1. 从点云更新高度图
    planner.update_heightmap_from_pointcloud(point_cloud)
    
    # 2. 规划放置位置
    result = planner.plan_placement(item_L, item_W, item_H)
    
    if result is None:
        print("无法放置当前货物。")
        return None
    
    # 3. 输出结果
    print(f"放置方案:")
    print(f"  {format_pose_string(result['pose'])}")
    
    return result


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="3D视觉装箱系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                        # 普通模拟演示
  python main.py --xy-only              # 仅XY旋转模式演示
  python main.py --ply scene.ply        # 从PLY文件加载初始状态
  python main.py --ply scene.ply --xy-only  # PLY + 仅XY旋转
  python main.py --uneven               # 不平底面演示
        """
    )
    parser.add_argument('--ply', type=str, default=None,
                        help='PLY 文件路径，用于加载初始笼内点云状态')
    parser.add_argument('--xy-only', action='store_true', default=False,
                        help='启用仅XY旋转模式（物体底面朝下，绕Z轴旋转0°/90°）')
    parser.add_argument('--uneven', action='store_true', default=False,
                        help='运行不平底面装箱演示')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.ply:
        run_demo_ply(args.ply, xy_only=args.xy_only)
    elif args.uneven:
        run_demo_uneven_surface(xy_only=args.xy_only)
    else:
        run_demo(xy_only=args.xy_only)
