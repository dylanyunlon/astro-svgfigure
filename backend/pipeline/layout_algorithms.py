"""
layout_algorithms.py — Algorithmic core for layout processing.
Dense computation. No API calls. No I/O. Pure algorithms.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

# §1 IoU + NMS ──────────────────────────────────────────────────────────

def iou(a: Dict, b: Dict) -> float:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0

def nms(elements: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Non-maximum suppression. Sort by area desc, greedily suppress."""
    s = sorted(elements, key=lambda e: e.get("bbox",{}).get("width",0)*e.get("bbox",{}).get("height",0), reverse=True)
    keep, suppressed = [], set()
    for i, ei in enumerate(s):
        if i in suppressed: continue
        keep.append(ei)
        bi = ei.get("bbox", {})
        for j in range(i+1, len(s)):
            if j not in suppressed and iou(bi, s[j].get("bbox", {})) > iou_threshold:
                suppressed.add(j)
    return keep

# §2 Hungarian matching ──────────────────────────────────────────────────

def _bbox_distance(a: Dict, b: Dict) -> float:
    acx, acy = a["x"]+a["width"]/2, a["y"]+a["height"]/2
    bcx, bcy = b["x"]+b["width"]/2, b["y"]+b["height"]/2
    return math.sqrt((acx-bcx)**2+(acy-bcy)**2) + 0.5*math.sqrt((a["width"]-b["width"])**2+(a["height"]-b["height"])**2)

def hungarian_match(frame_a: List[Dict], frame_b: List[Dict], max_distance: float = 50.0) -> List[Tuple[int,int,float]]:
    """Optimal bipartite matching. scipy O(n³) with greedy fallback."""
    n, m = len(frame_a), len(frame_b)
    if n == 0 or m == 0: return []
    INF = max_distance * 10
    size = max(n, m)
    cost = [[INF]*size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            d = _bbox_distance(frame_a[i].get("bbox",{}), frame_b[j].get("bbox",{}))
            cost[i][j] = d if d < max_distance else INF
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
        return [(int(r),int(c),cost[r][c]) for r,c in zip(ri,ci) if r<n and c<m and cost[r][c]<max_distance]
    except ImportError: pass
    # Greedy fallback
    pairs = sorted([(cost[i][j],i,j) for i in range(n) for j in range(m) if cost[i][j]<max_distance])
    used_a, used_b, matches = set(), set(), []
    for d,i,j in pairs:
        if i not in used_a and j not in used_b:
            matches.append((i,j,d)); used_a.add(i); used_b.add(j)
    return matches

# §3 Union-Find ──────────────────────────────────────────────────────────

class UnionFind:
    __slots__ = ("parent","rank","size")
    def __init__(self, n: int):
        self.parent = list(range(n)); self.rank = [0]*n; self.size = [1]*n
    def find(self, x: int) -> int:
        r = x
        while self.parent[r] != r: r = self.parent[r]
        while self.parent[x] != r: self.parent[x], x = r, self.parent[x]
        return r
    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra; self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1
        return True
    def groups(self) -> Dict[int, List[int]]:
        g: Dict[int,List[int]] = {}
        for i in range(len(self.parent)): g.setdefault(self.find(i),[]).append(i)
        return g

def group_elements_across_frames(layouts: List[List[Dict]], max_distance: float = 30.0):
    """Hungarian per frame-pair → Union-Find merge → groups."""
    flat = []
    offsets = []
    for fi, layout in enumerate(layouts):
        offsets.append(len(flat))
        for elem in layout: flat.append((fi, elem))
    n = len(flat)
    if n == 0: return [], []
    uf = UnionFind(n)
    for fi in range(len(layouts)):
        for fj in range(fi+1, len(layouts)):
            for ai, bj, _ in hungarian_match(layouts[fi], layouts[fj], max_distance):
                uf.union(offsets[fi]+ai, offsets[fj]+bj)
    return list(uf.groups().values()), flat

# §4 Grid inference + snapping ───────────────────────────────────────────

def infer_grid(elements: List[Dict], candidates: Tuple[int,...] = (2,4,8)) -> int:
    """Detect grid from data: histogram coordinate residues mod candidate."""
    coords = []
    for e in elements:
        b = e.get("bbox",{})
        coords.extend([int(b.get("x",0)), int(b.get("y",0)), int(b.get("width",0)), int(b.get("height",0))])
    if not coords: return candidates[0]
    best_grid, best_ratio = candidates[0], -1.0
    for g in candidates:
        ratio = sum(1 for c in coords if c%g==0) / len(coords)
        if ratio > best_ratio or (ratio == best_ratio and g > best_grid):
            best_ratio, best_grid = ratio, g
    return best_grid

def snap_to_grid(val: int, grid: int) -> int:
    return round(val/grid)*grid if grid > 1 else val

_STD = (16,20,24,28,32,36,40,44,48,52,56,64,72,80,96,128,160,192,256)

def snap_dimension(val: int, tolerance: int = 3) -> int:
    """Binary search nearest standard UI size."""
    if val <= 0: return val
    lo, hi, best, best_d = 0, len(_STD)-1, val, tolerance+1
    while lo <= hi:
        mid = (lo+hi)//2
        d = abs(_STD[mid]-val)
        if d < best_d: best_d, best = d, _STD[mid]
        if _STD[mid] < val: lo = mid+1
        elif _STD[mid] > val: hi = mid-1
        else: return _STD[mid]
    return best if best_d <= tolerance else val

# §5 DoubleBuffer (CCCL port) ───────────────────────────────────────────

class DoubleBuffer:
    __slots__ = ("_s","selector")
    def __init__(self, a=None, b=None): self._s = [a,b]; self.selector = 0
    def current(self): return self._s[self.selector]
    def alternate(self): return self._s[self.selector^1]
    def set_current(self, v): self._s[self.selector] = v
    def set_alternate(self, v): self._s[self.selector^1] = v
    def swap(self): self.selector ^= 1

# §6 Spatial bucket index ───────────────────────────────────────────────

class SpatialIndex:
    __slots__ = ("cell","buckets")
    def __init__(self, cell: int = 64): self.cell = cell; self.buckets: Dict[Tuple[int,int],List[int]] = {}
    def build(self, elements: List[Dict]):
        self.buckets.clear()
        for i, e in enumerate(elements):
            b = e.get("bbox",{})
            k = (int(b.get("x",0))//self.cell, int(b.get("y",0))//self.cell)
            self.buckets.setdefault(k,[]).append(i)
    def query(self, bbox: Dict) -> List[int]:
        cx, cy = int(bbox.get("x",0))//self.cell, int(bbox.get("y",0))//self.cell
        r = []
        for dx in (-1,0,1):
            for dy in (-1,0,1): r.extend(self.buckets.get((cx+dx,cy+dy),[]))
        return r
