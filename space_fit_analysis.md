# 空间匹配度策略分析 — P1 优化 vs 新增 P0

## 1. 当前 P1 (void_volume) 到底在做什么

```python
void_heights = place_height - region        # 物品底面与下方表面的高度差
void_volume = sum(max(0, void_heights)) * r^2   # 积分得到底部"悬空气泡"体积
```

**物理含义**：测量物品底面与支撑表面之间的"空气间隙"总体积。

```
      +──────────────+ ← place_height (物品底面)
      |  ████████████|
      |  ████ void ██| ← 这些空隙就是 void_volume
      |  ██████ █████|
   ───+──+───+──+────+─── ← 高度图表面 (不平)
         |已有|  |已有|
```

**它能做到的**：
- 倾向选择表面平坦的位置（void=0 最好）
- 避免在阶梯状表面上方悬空放置

**它做不到的**：
- 不区分"小物品放大空间" vs "小物品放小空间"
- 地面放置时所有位置 void=0，完全无区分力

## 2. 你提出的"空间匹配度"到底在解决什么

**核心思想**（类似内存分配的 Best-Fit 策略）：

> 评估候选位置处"可用空间体积"与"物品体积"的匹配程度。
> 优先选择刚好能容纳物品的紧凑空间，将大空间留给未来的大物品。

```
场景：空笼底面，要放一个 10x10x10cm 的小物品

位置A：笼子里侧角落，左/后墙壁+右边已有物品 -> 可用空间约 12x15x120cm
  fit_ratio = (10x10x10) / (12x15x120) = 0.046  <- 太浪费了

位置B：两个已有物品之间的窄缝 -> 可用空间约 12x10x50cm
  fit_ratio = (10x10x10) / (12x10x50) = 0.167   <- 更合适

位置C：恰好有个小凹槽 -> 可用空间约 11x11x12cm
  fit_ratio = (10x10x10) / (11x11x12) = 0.689   <- 最佳匹配！
```

## 3. 对比分析

| 维度 | P1: void_volume | 空间匹配度 (fit_ratio) |
|------|----------------|----------------------|
| **测量对象** | 物品底面下方的纵向间隙 | 候选位置处可用的最大长方体空间 |
| **物理含义** | "放下去之后底面贴合程度" | "这个空间是不是为这个物品量身定做的" |
| **空笼区分力** | 全部 void=0 无法区分 | 角落空间小(ratio高) vs 中间空间大(ratio低) |
| **堆叠层区分力** | 不平表面有区分 | 上层凹槽空间更紧凑 |
| **计算复杂度** | O(item_area) 极快 | O(heightmap) 需要扩展搜索 |
| **策略层级** | 战术级（表面质量） | 战略级（空间分配） |

## 4. 结论：应添加为新的 P0，而非替换 P1

**理由**：

1. **两者测量维度正交**：void_volume 测量纵向贴合，fit_ratio 测量整体空间匹配，互不替代
2. **策略层次不同**：空间匹配是更高层级的决策（"放哪个区域"），void_volume 是更低层级的决策（"区域内哪个精确位置"）
3. **保留 void_volume 的安全价值**：在同样紧凑的空间中，void_volume 仍能区分平坦 vs 不平坦表面

**新排序策略**：

```
(P0: -fit_ratio,  P1: void_volume,  P2: -adjacency,  P3: z_max,  P4: corner_dist)
  ^  空间匹配度    ^  底部空隙       ^  贴合度        ^  高度     ^  里左角距离
  越大越好(取负)   越小越好         越大越好(取负)   越小越好   越小越好
```

## 5. 可用最大长方体空间的计算方法

核心问题：给定候选位置 (row, col) 和放置高度 place_height，如何高效计算该处的"最大可用长方体空间"？

### 算法：从物品占位区向四方向扩展

```python
def _compute_max_available_volume(self, hm, row, col, item_rows, item_cols,
                                   place_height, up_dim):
    """
    计算候选位置处可容纳的最大长方体可用空间体积。

    方法：以物品占位区为种子，沿 4 个方向逐步扩展，
    直到碰到高于 place_height 的障碍物或笼壁。
    可用高度 = cage_height - place_height。
    """
    total_rows, total_cols = hm.shape
    top_z = place_height  # 可用空间的底面高度

    # 向上可用高度
    available_height = self.processor.cage_height - top_z
    if available_height <= 0:
        return 0.0

    # 从物品占位区开始，向四个方向扩展寻找同层空白区边界
    # "同层" = 该 cell 的高度 <= place_height（物品可以坐在上面）

    # 向左扩展
    left = col
    while left > 0 and np.all(hm[row:row+item_rows, left-1] <= top_z + 0.005):
        left -= 1

    # 向右扩展
    right = col + item_cols
    while right < total_cols and np.all(hm[row:row+item_rows, right] <= top_z + 0.005):
        right += 1

    # 向外扩展（小 row 方向）
    front = row
    while front > 0 and np.all(hm[front-1, left:right] <= top_z + 0.005):
        front -= 1

    # 向里扩展（大 row 方向）
    back = row + item_rows
    while back < total_rows and np.all(hm[back, left:right] <= top_z + 0.005):
        back += 1

    expanded_width  = (right - left) * self.processor.resolution
    expanded_depth  = (back - front) * self.processor.resolution

    return expanded_width * expanded_depth * available_height
```

### fit_ratio 计算

```python
item_volume = base_x * base_y * up_dim
available_volume = _compute_max_available_volume(...)
fit_ratio = item_volume / available_volume  if available_volume > 0  else 1.0
# fit_ratio in (0, 1]，越接近 1 越好 -> 取负后越小越好
```

> [!TIP]
> fit_ratio 应该做适当的分桶圆整（如保留 2 位小数），避免微小的浮点差异
> 过度压制后续的 P1-P4 排序维度。
