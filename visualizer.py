"""
3D 装箱系统可视化模块。
使用 Matplotlib 和 Open3D 展示高度图、俯视图和 3D 装箱结果。
"""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def _configure_matplotlib_fonts():
    """尽量选择常见中文字体，避免标题和坐标轴乱码。"""
    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = [font for font in candidate_fonts if font in available]
    if selected:
        plt.rcParams["font.sans-serif"] = selected + list(
            plt.rcParams.get("font.sans-serif", [])
        )
    plt.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_fonts()


COLORS = [
    [0.85, 0.33, 0.33],
    [0.33, 0.65, 0.85],
    [0.33, 0.85, 0.45],
    [0.85, 0.75, 0.33],
    [0.65, 0.33, 0.85],
    [0.85, 0.55, 0.33],
    [0.33, 0.85, 0.85],
    [0.85, 0.33, 0.65],
]


def visualize_heightmap(heightmap: np.ndarray,
                        title: str = "高度图",
                        save_path: Optional[str] = None):
    """用 Matplotlib 热力图显示高度图。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    im = ax.imshow(
        heightmap,
        cmap="YlOrRd",
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel("X（宽度方向，左到右）")
    ax.set_ylabel("Y（深度方向，外到里）")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("高度（m）")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"高度图已保存: {save_path}")

    plt.show()


def visualize_fitted_heightmap(raw_heightmap: np.ndarray,
                               fitted_heightmap: np.ndarray,
                               plane_label_map: Optional[np.ndarray] = None,
                               title: str = "拟合高度图对比",
                               save_path: Optional[str] = None):
    """显示原始高度图、拟合后高度图、差值图和可选的平面标签图。"""
    delta_heightmap = fitted_heightmap - raw_heightmap
    has_labels = plane_label_map is not None
    ncols = 4 if has_labels else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    vmin = min(float(np.min(raw_heightmap)), float(np.min(fitted_heightmap)))
    vmax = max(float(np.max(raw_heightmap)), float(np.max(fitted_heightmap)))

    im_raw = axes[0].imshow(
        raw_heightmap,
        cmap="YlOrRd",
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("原始高度图")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    im_fitted = axes[1].imshow(
        fitted_heightmap,
        cmap="YlOrRd",
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("拟合后高度图")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    cbar = plt.colorbar(im_fitted, ax=axes[:2], shrink=0.8)
    cbar.set_label("高度（m）")

    delta_abs_max = float(np.max(np.abs(delta_heightmap))) if delta_heightmap.size else 0.0
    if delta_abs_max <= 0.0:
        delta_abs_max = 1e-6
    im_delta = axes[2].imshow(
        delta_heightmap,
        cmap="coolwarm",
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )
    axes[2].set_title("拟合差值图")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    cbar_delta = plt.colorbar(im_delta, ax=axes[2], shrink=0.8)
    cbar_delta.set_label("拟合后 - 原始（m）")

    if has_labels:
        masked_labels = np.ma.masked_less(plane_label_map, 0)
        cmap = plt.cm.get_cmap("tab20").copy()
        cmap.set_bad(color="black", alpha=0.15)
        im_labels = axes[3].imshow(
            masked_labels,
            cmap=cmap,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
        )
        axes[3].set_title("平面标签图")
        axes[3].set_xlabel("X")
        axes[3].set_ylabel("Y")
        plt.colorbar(im_labels, ax=axes[3], shrink=0.8)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"拟合高度图对比已保存: {save_path}")

    plt.show()


def visualize_packing_2d(heightmap: np.ndarray,
                         placed_items: List[dict],
                         title: str = "装箱俯视图",
                         save_path: Optional[str] = None):
    """在高度图上叠加已放置货物的底面轮廓。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    im = ax.imshow(
        heightmap,
        cmap="Greys",
        origin="lower",
        aspect="equal",
        alpha=0.6,
        interpolation="nearest",
    )

    for i, item in enumerate(placed_items):
        result = item["result"]
        row, col = result["grid_pos"]
        rows, cols = result["item_grid_size"]
        color = COLORS[i % len(COLORS)]

        rect = plt.Rectangle(
            (col, row),
            cols,
            rows,
            linewidth=2,
            edgecolor=color,
            facecolor=color + [0.3],
        )
        ax.add_patch(rect)
        ax.text(
            col + cols / 2,
            row + rows / 2,
            f"#{i + 1}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    ax.set_xlabel("X（宽度方向，左到右）")
    ax.set_ylabel("Y（深度方向，外到里）")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("高度（m）")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"俯视图已保存: {save_path}")

    plt.show()


def create_cage_wireframe(origin, width, length, height, color=None):
    """创建笼体线框（Open3D LineSet）。"""
    if not HAS_OPEN3D:
        return None

    if color is None:
        color = [1.0, 0.5, 0.0]

    x0, y0, z0 = origin
    x1, y1, z1 = x0 + width, y0 + length, z0 + height

    points = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def create_box_mesh(center, half_extents, rotation_matrix, color):
    """创建一个旋转后的长方体网格。"""
    if not HAS_OPEN3D:
        return None

    mesh = o3d.geometry.TriangleMesh.create_box(
        width=half_extents[0] * 2,
        height=half_extents[1] * 2,
        depth=half_extents[2] * 2,
    )
    mesh.translate(-np.array(half_extents))
    mesh.rotate(rotation_matrix, center=np.zeros(3))
    mesh.translate(center)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def visualize_packing_3d(cage_origin: Tuple[float, float, float],
                         cage_width: float,
                         cage_length: float,
                         cage_height: float,
                         placed_items: List[dict],
                         point_cloud: Optional[np.ndarray] = None,
                         point_colors: Optional[np.ndarray] = None,
                         title: str = "3D装箱结果"):
    """使用 Open3D 展示 3D 装箱结果。"""
    if not HAS_OPEN3D:
        print("未安装 Open3D，跳过 3D 可视化。可使用 `pip install open3d` 安装。")
        return

    geometries = []

    cage = create_cage_wireframe(cage_origin, cage_width, cage_length, cage_height)
    if cage is not None:
        geometries.append(cage)

    if point_cloud is not None and len(point_cloud) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        if point_colors is not None and len(point_colors) == len(point_cloud):
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
        else:
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd)

    for i, item in enumerate(placed_items):
        result = item["result"]
        pose = result["pose"]
        orientation = result["orientation"]
        up_dim = orientation["up_dim"]
        orig_l, orig_w, orig_h = item["dimensions"]
        center = np.array(pose["position"])
        color = COLORS[i % len(COLORS)]

        simulated_pose = result.get("simulated_pose")
        if simulated_pose:
            orig_l, orig_w, orig_h = item["dimensions"]
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=orig_l,
                height=orig_w,
                depth=orig_h,
            )
            mesh.translate([-orig_l / 2, -orig_w / 2, -orig_h / 2])
            mesh.rotate(simulated_pose["rotation_matrix"], center=np.zeros(3))
            mesh.translate(simulated_pose["position"])
        else:
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=orig_l,
                height=orig_w,
                depth=orig_h,
            )
            mesh.translate([-orig_l / 2, -orig_w / 2, -orig_h / 2])
            mesh.rotate(pose["rotation_matrix"], center=np.zeros(3))
            mesh.translate(center)

        mesh.paint_uniform_color(color)
        mesh.compute_vertex_normals()
        geometries.append(mesh)

        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        marker_center = center + np.array([0.0, 0.0, up_dim / 2 + 0.02])
        marker.translate(marker_center)
        marker.paint_uniform_color(color)
        geometries.append(marker)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(coord)

    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"  已放置 {len(placed_items)} 个货物")
    print(f"{'=' * 50}")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=720,
    )
