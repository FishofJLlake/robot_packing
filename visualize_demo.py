"""
3D装箱系统 — 逐步可视化演示

使用 matplotlib 3D 绘图，逐步展示每个货物的放置过程。
每一步显示:
  - 左图: 3D视图（笼车 + 已放置货物 + 新放置货物高亮）
  - 右图: 高度图热力图（俯视图）
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
import os

from config import CAGE_WIDTH, CAGE_LENGTH, CAGE_HEIGHT, HEIGHTMAP_RESOLUTION, PLACEMENT_GAP
from point_cloud_processor import PointCloudProcessor
from stability_checker import StabilityChecker
from packing_planner import PackingPlanner
from pose_generator import format_pose_string


# ─── 颜色方案（更美观的配色） ───
ITEM_COLORS = [
    '#E74C3C',  # 红
    '#3498DB',  # 蓝
    '#2ECC71',  # 绿
    '#F39C12',  # 橙
    '#9B59B6',  # 紫
    '#1ABC9C',  # 青绿
    '#E67E22',  # 深橙
    '#E84393',  # 粉
    '#00B894',  # 薄荷绿
    '#6C5CE7',  # 靛蓝
    '#FDCB6E',  # 金黄
    '#74B9FF',  # 天蓝
]

CAGE_COLOR = '#E67E22'  # 笼车线框颜色
BG_COLOR   = '#1a1a2e'  # 背景色
GRID_COLOR = '#16213e'  # 网格色


def hex_to_rgb(hex_color):
    """Hex颜色转 [0,1] RGB"""
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]


def draw_box_faces(ax, origin, size, color, alpha=0.6, edge_color=None, linewidth=0.8):
    """
    在3D坐标轴上绘制一个长方体的六个面。
    
    Parameters
    ----------
    ax : Axes3D
    origin : (x0, y0, z0) 长方体最小角
    size : (dx, dy, dz) 各方向尺寸
    color : str or RGB
    alpha : float 透明度
    """
    x0, y0, z0 = origin
    dx, dy, dz = size
    
    if edge_color is None:
        edge_color = color
    
    # 八个顶点
    v = np.array([
        [x0,      y0,      z0],
        [x0+dx,   y0,      z0],
        [x0+dx,   y0+dy,   z0],
        [x0,      y0+dy,   z0],
        [x0,      y0,      z0+dz],
        [x0+dx,   y0,      z0+dz],
        [x0+dx,   y0+dy,   z0+dz],
        [x0,      y0+dy,   z0+dz],
    ])
    
    # 六个面（按顶点索引）
    faces = [
        [v[0], v[1], v[5], v[4]],  # 前面 (y=y0)
        [v[2], v[3], v[7], v[6]],  # 后面 (y=y0+dy)
        [v[0], v[3], v[7], v[4]],  # 左面 (x=x0)
        [v[1], v[2], v[6], v[5]],  # 右面 (x=x0+dx)
        [v[0], v[1], v[2], v[3]],  # 底面 (z=z0)
        [v[4], v[5], v[6], v[7]],  # 顶面 (z=z0+dz)
    ]
    
    rgb = hex_to_rgb(color) if isinstance(color, str) else color
    
    collection = Poly3DCollection(
        faces,
        alpha=alpha,
        facecolor=rgb,
        edgecolor=hex_to_rgb(edge_color) if isinstance(edge_color, str) else edge_color,
        linewidth=linewidth,
    )
    ax.add_collection3d(collection)


def draw_cage_wireframe(ax, origin, width, length, height):
    """绘制笼车线框。"""
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + width, y0 + length, z0 + height
    
    # 底面
    bottom = [[x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0]]
    # 顶面
    top = [[x0,y0,z1], [x1,y0,z1], [x1,y1,z1], [x0,y1,z1]]
    
    cage_rgb = hex_to_rgb(CAGE_COLOR)
    
    # 底面线
    for i in range(4):
        j = (i + 1) % 4
        ax.plot3D(*zip(bottom[i], bottom[j]), color=cage_rgb, linewidth=2.0, alpha=0.9)
    # 顶面线
    for i in range(4):
        j = (i + 1) % 4
        ax.plot3D(*zip(top[i], top[j]), color=cage_rgb, linewidth=2.0, alpha=0.9)
    # 竖边
    for i in range(4):
        ax.plot3D(*zip(bottom[i], top[i]), color=cage_rgb, linewidth=2.0, alpha=0.9)
    
    # 半透明底面
    bottom_face = Poly3DCollection(
        [bottom], alpha=0.08, facecolor=cage_rgb, edgecolor='none'
    )
    ax.add_collection3d(bottom_face)


def create_step_figure(step, total_steps, placed_items_so_far, current_item_info,
                       heightmap, cage_origin, cage_w, cage_l, cage_h, resolution,
                       save_path=None):
    """
    创建单步可视化图。
    
    Parameters
    ----------
    step : int 当前步骤编号 (1-indexed)
    total_steps : int 总步骤数
    placed_items_so_far : list 已放置的货物列表
    current_item_info : dict 当前放置的货物信息（高亮显示）
    heightmap : ndarray 当前高度图
    """
    fig = plt.figure(figsize=(18, 8), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1], wspace=0.15)
    
    # ─── 左图: 3D视图 ───
    ax3d = fig.add_subplot(gs[0], projection='3d', facecolor=BG_COLOR)
    
    # 设置3D视角
    ax3d.view_init(elev=25, azim=-55)
    
    # 绘制笼车
    draw_cage_wireframe(ax3d, cage_origin, cage_w, cage_l, cage_h)
    
    # 绘制已放置的货物（之前放的，较暗）
    for i, item in enumerate(placed_items_so_far[:-1] if placed_items_so_far else []):
        result = item['result']
        ori = result['orientation']
        base_x, base_y = ori['base_dims']
        up_dim = ori['up_dim']
        row, col = result['grid_pos']
        place_h = result['place_height']
        
        box_origin = (
            cage_origin[0] + col * resolution,
            cage_origin[1] + row * resolution,
            cage_origin[2] + place_h,
        )
        box_size = (base_x, base_y, up_dim)
        
        color = ITEM_COLORS[i % len(ITEM_COLORS)]
        draw_box_faces(ax3d, box_origin, box_size, color, alpha=0.35, linewidth=0.5)
    
    # 绘制当前放置的货物（高亮，较亮）
    if placed_items_so_far:
        item = placed_items_so_far[-1]
        result = item['result']
        ori = result['orientation']
        base_x, base_y = ori['base_dims']
        up_dim = ori['up_dim']
        row, col = result['grid_pos']
        place_h = result['place_height']
        
        box_origin = (
            cage_origin[0] + col * resolution,
            cage_origin[1] + row * resolution,
            cage_origin[2] + place_h,
        )
        box_size = (base_x, base_y, up_dim)
        
        idx = len(placed_items_so_far) - 1
        color = ITEM_COLORS[idx % len(ITEM_COLORS)]
        # 高亮当前货物（更不透明 + 白色边框）
        draw_box_faces(ax3d, box_origin, box_size, color, 
                      alpha=0.85, edge_color='#FFFFFF', linewidth=1.5)
    
    # 坐标轴设置
    margin = 0.05
    ax3d.set_xlim(cage_origin[0] - margin, cage_origin[0] + cage_w + margin)
    ax3d.set_ylim(cage_origin[1] - margin, cage_origin[1] + cage_l + margin)
    ax3d.set_zlim(cage_origin[2] - margin, cage_origin[2] + cage_h + margin)
    
    ax3d.set_xlabel('X (宽度)', color='white', fontsize=9, labelpad=8)
    ax3d.set_ylabel('Y (深度)', color='white', fontsize=9, labelpad=8)
    ax3d.set_zlabel('Z (高度)', color='white', fontsize=9, labelpad=8)
    
    # 设置轴标签颜色
    ax3d.tick_params(colors='white', labelsize=7)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor('#333355')
    ax3d.yaxis.pane.set_edgecolor('#333355')
    ax3d.zaxis.pane.set_edgecolor('#333355')
    ax3d.grid(True, alpha=0.15, color='#8888aa')
    
    # ─── 中图: 高度图 ───
    ax_hm = fig.add_subplot(gs[1], facecolor=BG_COLOR)
    
    im = ax_hm.imshow(
        heightmap, cmap='magma', origin='lower',
        aspect='equal', interpolation='nearest',
        vmin=0, vmax=cage_h
    )
    
    # 叠加货物轮廓
    for i, item in enumerate(placed_items_so_far):
        result = item['result']
        row, col = result['grid_pos']
        rows, cols = result['item_grid_size']
        color = ITEM_COLORS[i % len(ITEM_COLORS)]
        
        is_current = (i == len(placed_items_so_far) - 1)
        lw = 2.5 if is_current else 1.0
        ls = '-' if is_current else '--'
        
        rect = Rectangle(
            (col - 0.5, row - 0.5), cols, rows,
            linewidth=lw, linestyle=ls,
            edgecolor=color, facecolor='none',
        )
        ax_hm.add_patch(rect)
        
        ax_hm.text(
            col + cols / 2, row + rows / 2,
            f'{i+1}', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white',
        )
    
    ax_hm.set_xlabel('X', color='white', fontsize=9)
    ax_hm.set_ylabel('Y', color='white', fontsize=9)
    ax_hm.set_title('高度图', color='white', fontsize=11, fontweight='bold')
    ax_hm.tick_params(colors='white', labelsize=7)
    
    cbar = plt.colorbar(im, ax=ax_hm, shrink=0.7, pad=0.02)
    cbar.set_label('高度 (m)', color='white', fontsize=8)
    cbar.ax.tick_params(colors='white', labelsize=7)
    
    # ─── 右图: 信息面板 ───
    ax_info = fig.add_subplot(gs[2], facecolor=BG_COLOR)
    ax_info.axis('off')
    
    info_lines = []
    info_lines.append(f"步骤 {step} / {total_steps}")
    info_lines.append(f"{'─' * 24}")
    
    if current_item_info:
        L, W, H = current_item_info['dimensions']
        result = current_item_info['result']
        pose = result['pose']
        pos = pose['position']
        euler = pose['orientation_euler']
        
        info_lines.append(f"货物尺寸:")
        info_lines.append(f"  {L:.3f} × {W:.3f} × {H:.3f} m")
        info_lines.append(f"")
        info_lines.append(f"朝向: {result['orientation']['desc']}")
        info_lines.append(f"")
        info_lines.append(f"位置 (x, y, z):")
        info_lines.append(f"  ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        info_lines.append(f"")
        info_lines.append(f"欧拉角 (R, P, Y):")
        info_lines.append(f"  ({np.rad2deg(euler[0]):.1f}°, "
                         f"{np.rad2deg(euler[1]):.1f}°, "
                         f"{np.rad2deg(euler[2]):.1f}°)")
        info_lines.append(f"")
        info_lines.append(f"放置高度: {result['place_height']:.3f} m")
        info_lines.append(f"支撑率: {result['stability']['support_ratio']:.0%}")
        info_lines.append(f"评分: {result['score']:.2f}")
        info_lines.append(f"")
        info_lines.append(f"{'─' * 24}")
        
        # 统计
        total_vol = cage_w * cage_l * cage_h
        item_vol = sum(
            it['dimensions'][0] * it['dimensions'][1] * it['dimensions'][2]
            for it in placed_items_so_far
        )
        info_lines.append(f"已放置: {len(placed_items_so_far)} 件")
        info_lines.append(f"体积利用率: {item_vol/total_vol:.1%}")
        info_lines.append(f"最高点: {np.max(heightmap):.3f} m")
    
    info_text = '\n'.join(info_lines)
    ax_info.text(
        0.05, 0.95, info_text,
        transform=ax_info.transAxes,
        fontsize=10, fontfamily='sans-serif',
        verticalalignment='top',
        color='#ECF0F1',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='#2c2c54',
            edgecolor='#474787',
            alpha=0.9,
        ),
    )
    
    # 总标题
    fig.suptitle(
        f'3D 装箱过程可视化  ·  步骤 {step}/{total_steps}',
        fontsize=16, fontweight='bold', color='white',
        y=0.98,
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
    
    return fig


def run_visual_demo():
    """运行逐步可视化装箱演示。"""
    
    print("=" * 60)
    print("  3D装箱过程 — 逐步可视化演示")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建系统
    cage_origin = (0.0, 0.0, 0.0)
    processor = PointCloudProcessor(
        cage_origin=cage_origin,
        cage_width=CAGE_WIDTH,
        cage_length=CAGE_LENGTH,
        cage_height=CAGE_HEIGHT,
        resolution=HEIGHTMAP_RESOLUTION,
    )
    checker = StabilityChecker()
    planner = PackingPlanner(processor, checker)
    
    # 模拟来料
    items = [
        (0.40, 0.30, 0.25),   # #1 中等快递箱
        (0.35, 0.25, 0.20),   # #2 较小快递箱
        (0.50, 0.35, 0.30),   # #3 大号快递箱
        (0.30, 0.25, 0.15),   # #4 小号快递箱
        (0.45, 0.30, 0.25),   # #5 中等快递箱
        (0.25, 0.20, 0.15),   # #6 迷你快递箱
        (0.40, 0.35, 0.30),   # #7 中大号快递箱
        (0.35, 0.30, 0.20),   # #8 中等快递箱
        (0.30, 0.20, 0.15),   # #9 小号
        (0.45, 0.35, 0.25),   # #10 中大号
        (0.25, 0.25, 0.20),   # #11 正方形小箱
        (0.50, 0.40, 0.30),   # #12 超大号
    ]
    
    total = len(items)
    placed_so_far = []
    figures = []
    
    for i, (L, W, H) in enumerate(items):
        step = i + 1
        print(f"\n  步骤 {step}/{total}: 放置货物 {L:.3f}×{W:.3f}×{H:.3f} m ...", end=' ')
        
        result = planner.plan_placement(L, W, H)
        
        if result is None:
            print("❌ 无法放置")
            continue
        
        # 更新高度图
        row, col = result['grid_pos']
        item_rows, item_cols = result['item_grid_size']
        planner.update_heightmap_with_placement(
            row, col, item_rows, item_cols,
            result['place_height'], result['orientation']['up_dim']
        )
        
        placed_so_far.append({
            'dimensions': (L, W, H),
            'result': result,
        })
        
        print(f"✅ pos=({result['pose']['position'][0]:.3f}, "
              f"{result['pose']['position'][1]:.3f}, "
              f"{result['pose']['position'][2]:.3f})")
        
        # 创建可视化
        save_path = os.path.join(output_dir, f'step_{step:02d}.png')
        fig = create_step_figure(
            step=step,
            total_steps=total,
            placed_items_so_far=placed_so_far,
            current_item_info=placed_so_far[-1],
            heightmap=planner.heightmap.copy(),
            cage_origin=cage_origin,
            cage_w=CAGE_WIDTH,
            cage_l=CAGE_LENGTH,
            cage_h=CAGE_HEIGHT,
            resolution=HEIGHTMAP_RESOLUTION,
            save_path=save_path,
        )
        figures.append(fig)
        print(f"     图片已保存: {save_path}")
    
    # 统计
    stats = planner.get_packing_stats()
    print(f"\n{'=' * 60}")
    print(f"  装箱完成！")
    print(f"  成功放置: {stats['num_items']}/{total} 个")
    print(f"  体积利用率: {stats['volume_utilization']:.1%}")
    print(f"  最大高度: {stats['max_height']:.3f} m")
    print(f"  图片保存目录: {output_dir}")
    print(f"{'=' * 60}")
    
    # 显示所有步骤的缩略图总览
    create_overview_figure(placed_so_far, planner, cage_origin, output_dir)
    
    # 图片生成完毕
    print("\n  图片已全部生成并保存至 output/ 目录。")


def create_overview_figure(placed_items, planner, cage_origin, output_dir):
    """创建总览图：所有步骤的3D视图+最终高度图。"""
    
    n = len(placed_items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig = plt.figure(figsize=(5 * cols, 5 * rows + 2), facecolor=BG_COLOR)
    
    fig.suptitle(
        f'3D装箱全过程总览  ·  共 {n} 步',
        fontsize=18, fontweight='bold', color='white', y=0.98
    )
    
    resolution = planner.processor.resolution
    cage_w = planner.processor.cage_width
    cage_l = planner.processor.cage_length
    cage_h = planner.processor.cage_height
    
    # 重新模拟以获取每步的高度图快照
    temp_processor = PointCloudProcessor(
        cage_origin=cage_origin, cage_width=cage_w,
        cage_length=cage_l, cage_height=cage_h, resolution=resolution,
    )
    temp_hm = np.zeros((temp_processor.grid_rows, temp_processor.grid_cols))
    
    for idx, item in enumerate(placed_items):
        result = item['result']
        row, col = result['grid_pos']
        ir, ic = result['item_grid_size']
        up_dim = result['orientation']['up_dim']
        ph = result['place_height']
        
        new_h = ph + up_dim
        r_end = min(row + ir, temp_processor.grid_rows)
        c_end = min(col + ic, temp_processor.grid_cols)
        temp_hm[row:r_end, col:c_end] = np.maximum(
            temp_hm[row:r_end, col:c_end], new_h
        )
        
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d', facecolor=BG_COLOR)
        ax.view_init(elev=28, azim=-55)
        
        # 绘制笼车
        draw_cage_wireframe(ax, cage_origin, cage_w, cage_l, cage_h)
        
        # 绘制到这一步为止的所有货物
        for j in range(idx + 1):
            it = placed_items[j]
            r = it['result']
            ori = r['orientation']
            bx, by = ori['base_dims']
            ud = ori['up_dim']
            rr, cc = r['grid_pos']
            p_h = r['place_height']
            
            box_origin = (
                cage_origin[0] + cc * resolution,
                cage_origin[1] + rr * resolution,
                cage_origin[2] + p_h,
            )
            
            color = ITEM_COLORS[j % len(ITEM_COLORS)]
            is_new = (j == idx)
            alpha = 0.85 if is_new else 0.4
            edge = '#FFFFFF' if is_new else color
            lw = 1.5 if is_new else 0.5
            
            draw_box_faces(ax, box_origin, (bx, by, ud), color,
                          alpha=alpha, edge_color=edge, linewidth=lw)
        
        ax.set_xlim(-0.05, cage_w + 0.05)
        ax.set_ylim(-0.05, cage_l + 0.05)
        ax.set_zlim(-0.05, cage_h + 0.05)
        ax.set_title(f'步骤 {idx+1}', color='white', fontsize=11, fontweight='bold', pad=2)
        ax.tick_params(colors='white', labelsize=5)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333355')
        ax.yaxis.pane.set_edgecolor('#333355')
        ax.zaxis.pane.set_edgecolor('#333355')
        ax.grid(True, alpha=0.1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(output_dir, 'overview.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n  总览图已保存: {save_path}")


if __name__ == '__main__':
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    run_visual_demo()
