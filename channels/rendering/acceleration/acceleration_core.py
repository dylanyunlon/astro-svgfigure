import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)




# ---------------------------------------------------------------------------
# Internal geometry primitive
# ---------------------------------------------------------------------------

class _AABB2:
    """2-D axis-aligned bounding box (min_x, min_y, max_x, max_y)."""

    __slots__ = ("min_x", "min_y", "max_x", "max_y")

    def __init__(self,
                 min_x: float = float("inf"),  min_y: float = float("inf"),
                 max_x: float = float("-inf"), max_y: float = float("-inf")) -> None:
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    # half-perimeter — 2-D analogue of SAH surface area
    def cost(self) -> float:
        dx = self.max_x - self.min_x
        dy = self.max_y - self.min_y
        if dx < 0 or dy < 0:
            return 0.0
        return dx + dy

    def union(self, other: "_AABB2") -> "_AABB2":
        return _AABB2(
            min(self.min_x, other.min_x), min(self.min_y, other.min_y),
            max(self.max_x, other.max_x), max(self.max_y, other.max_y),
        )

    def overlaps(self, other: "_AABB2") -> bool:
        return (self.max_x >= other.min_x and other.max_x >= self.min_x and
                self.max_y >= other.min_y and other.max_y >= self.min_y)

    def contains(self, other: "_AABB2") -> bool:
        return (self.min_x <= other.min_x and self.min_y <= other.min_y and
                self.max_x >= other.max_x and self.max_y >= other.max_y)

    @staticmethod
    def from_bbox(bbox: dict) -> "_AABB2":
        """Accept the project's standard bbox dict {x, y, w, h} or {min, max}."""
        if "min" in bbox and "max" in bbox:
            mn, mx = bbox["min"], bbox["max"]
            return _AABB2(mn[0], mn[1], mx[0], mx[1])
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        return _AABB2(x, y, x + w, y + h)


# ---------------------------------------------------------------------------
# BVH node  (mirrors FDynamicBVH::FNode with MaxChildren = 4)
# ---------------------------------------------------------------------------

_MAX_CHILDREN = 4          # FDynamicBVH<MaxChildren=4>
_CHILD_MASK   = _MAX_CHILDREN - 1   # 0b11
_INDEX_SHIFT  = 2                   # log2(4)





class _BVHNode:
    """
    Internal BVH node holding up to _MAX_CHILDREN child slots.

    child_bounds[i]  : _AABB2 of the i-th child subtree / leaf
    child_ptr[i]     : encoded pointer
                         odd  → leaf:  cell_index = child_ptr[i] >> 1
                         even → inner: node_index  = child_ptr[i]
    parent_idx       : node array index of parent (None = root)
    """

    __slots__ = ("parent_idx", "num_children", "child_bounds", "child_ptr")

    def __init__(self) -> None:
        self.parent_idx:   Optional[int] = None
        self.num_children: int           = 0
        self.child_bounds: List[Optional[_AABB2]] = [None] * _MAX_CHILDREN
        self.child_ptr:    List[int]               = [0]   * _MAX_CHILDREN

    def is_full(self) -> bool:
        return self.num_children == _MAX_CHILDREN

    def is_root(self) -> bool:
        return self.parent_idx is None

    def union_bounds(self) -> _AABB2:
        b = _AABB2()
        for i in range(self.num_children):
            b = b.union(self.child_bounds[i])
        return b

    def get_bounds(self, slot: int) -> _AABB2:
        return self.child_bounds[slot & _CHILD_MASK]

    def get_ptr(self, slot: int) -> int:
        return self.child_ptr[slot & _CHILD_MASK]

    def is_leaf_slot(self, slot: int) -> bool:
        return bool(self.child_ptr[slot & _CHILD_MASK] & 1)


# ---------------------------------------------------------------------------
# AstroCellBVH — public API
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# AstroCellBVH — public API
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# AstroCellBVH — public API
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# AstroCellBVH — public API
# ---------------------------------------------------------------------------

class AstroCellBVH:
    """
    2-D dynamic AABB BVH for AstroCell overlap acceleration.

    Ports FDynamicBVH<4> from Renderer-Private/DynamicBVH.h:
      - Insert  → insert_cell
      - Remove  → remove_cell
      - Update  → update_cell_bounds
      - ForAll  → query_overlapping_cells

    All structural operations run in amortised O(log N) time;
    query_overlapping_cells is O(log N + k) where k = number of hits.
    """

    def __init__(self) -> None:
        # Node pool (index → _BVHNode); mirrors TArray<FNode> Nodes
        self._nodes:  List[_BVHNode] = []
        # Free-list head (node pool index), mirrors FreeHead
        self._free_head: Optional[int] = None
        # Leaf table: cell_id → encoded node-slot  (mirrors TArray<uint32> Leaves)
        self._leaves: Dict[str, int] = {}
        # Root node index (pool index) — None when tree is empty
        self._root: Optional[int] = None
        # Root AABB (mirrors FSingleRoot::Root.Bounds)
        self._root_bounds: _AABB2 = _AABB2()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def insert_cell(self, cell_id: str, bbox: dict) -> None:
        """
        Insert a cell into the BVH.
        Mirrors FDynamicBVH::Add(Bounds, Index).
        """
        aabb = _AABB2.from_bbox(bbox)
        if cell_id in self._leaves:
            # Already present — treat as update (mirrors check() + re-insert)
            self._extract(self._leaves[cell_id])
        leaf_ptr = self._encode_leaf(cell_id)

        if self._root is None:
            self._root = self._alloc_node()
            self._nodes[self._root].parent_idx = None
            self._nodes[self._root].num_children = 0

        slot = self._insert(aabb, leaf_ptr)
        self._leaves[cell_id] = slot
        self._root_bounds = self._root_bounds.union(aabb)

    def remove_cell(self, cell_id: str) -> None:
        """
        Remove a cell from the BVH.
        Mirrors FDynamicBVH::Remove(Index).
        """
        if cell_id not in self._leaves:
            return
        slot = self._leaves.pop(cell_id)
        self._extract(slot)
        # Recompute root bounds lazily from root node (mirrors Extract path)
        if self._root is not None:
            self._root_bounds = self._nodes[self._root].union_bounds()
        else:
            self._root_bounds = _AABB2()

    def update_cell_bounds(self, cell_id: str, bbox: dict) -> None:
        """
        Move a cell to new bounds.
        Mirrors FDynamicBVH::Update(Bounds, Index) = Remove + Add.
        """
        self.remove_cell(cell_id)
        self.insert_cell(cell_id, bbox)

    def query_overlapping_cells(self, bbox: dict) -> List[str]:
        """
        Return all cell_ids whose stored AABB overlaps *bbox*.
        Mirrors FDynamicBVH::ForAll(Bounds, Func) — O(log N + k).
        """
        if self._root is None:
            return []
        query_aabb = _AABB2.from_bbox(bbox)
        if not self._root_bounds.overlaps(query_aabb):
            return []

        results: List[str] = []
        stack:   List[int] = [self._root << _INDEX_SHIFT]

        while stack:
            node_slot = stack.pop()
            node = self._nodes[node_slot >> _INDEX_SHIFT]
            for i in range(node.num_children):
                cb = node.child_bounds[i]
                if cb is None or not cb.overlaps(query_aabb):
                    continue
                ptr = node.child_ptr[i]
                if ptr & 1:
                    # Leaf
                    cell_id = self._decode_leaf(ptr)
                    if cell_id is not None:
                        results.append(cell_id)
                else:
                    # Inner node
                    stack.append(ptr << _INDEX_SHIFT)

        return results

    def build_from_registry(self, cell_registry: dict) -> None:
        """
        Bulk-build the BVH from an existing cell registry dict.
        Mirrors FDynamicBVH::Build() with Morton-sort-based splitting.
        The registry format matches the one used by perform_nanite_visibility().
        """
        self._nodes.clear()
        self._free_head = None
        self._leaves.clear()
        self._root = None
        self._root_bounds = _AABB2()

        items: List[Tuple[str, _AABB2]] = []
        for cell_id, entry in cell_registry.items():
            bbox_data = entry.get("bbox", {})
            try:
                aabb = _AABB2.from_bbox(bbox_data)
            except (KeyError, TypeError, IndexError):
                continue
            items.append((cell_id, aabb))

        if not items:
            return

        # Compute scene-wide AABB for Morton normalisation
        scene = _AABB2()
        for _, ab in items:
            scene = scene.union(ab)
        sx = scene.max_x - scene.min_x or 1.0
        sy = scene.max_y - scene.min_y or 1.0

        # 2-D Morton codes: interleave 10-bit X and Y (mirrors FMortonArray ctor)
        def _morton2(x: float, y: float) -> int:
            xi = int(max(0.0, min(1023.0, x * 1023.0)))
            yi = int(max(0.0, min(1023.0, y * 1023.0)))
            # Spread bits: x in even positions, y in odd positions
            def _spread(v: int) -> int:
                v &= 0x3FF
                v = (v | (v << 8)) & 0x00FF00FF
                v = (v | (v << 4)) & 0x0F0F0F0F
                v = (v | (v << 2)) & 0x33333333
                v = (v | (v << 1)) & 0x55555555
                return v
            return _spread(xi) | (_spread(yi) << 1)

        sorted_items = sorted(
            items,
            key=lambda t: _morton2(
                ((t[1].min_x + t[1].max_x) * 0.5 - scene.min_x) / sx,
                ((t[1].min_y + t[1].max_y) * 0.5 - scene.min_y) / sy,
            ),
        )

        # Recursive BVH build via Morton-split (mirrors FDynamicBVH::Build stack)
        self._root = self._build_recursive(sorted_items, None)
        if self._root is not None:
            self._root_bounds = self._nodes[self._root].union_bounds()

    # ------------------------------------------------------------------
    # Internal helpers — mirrors private FDynamicBVH methods
    # ------------------------------------------------------------------

    # ── Leaf pointer encoding (odd = leaf, mirrors (Index << 1) | 1) ──

    # Map cell_id → integer index for pointer encoding
    _cell_to_idx: Dict[str, int] = {}
    _idx_to_cell: Dict[int, str] = {}
    _next_idx: int = 0

    def _encode_leaf(self, cell_id: str) -> int:
        if cell_id not in AstroCellBVH._cell_to_idx:
            idx = AstroCellBVH._next_idx
            AstroCellBVH._cell_to_idx[cell_id] = idx
            AstroCellBVH._idx_to_cell[idx] = cell_id
            AstroCellBVH._next_idx += 1
        return (AstroCellBVH._cell_to_idx[cell_id] << 1) | 1

    @staticmethod
    def _decode_leaf(ptr: int) -> Optional[str]:
        return AstroCellBVH._idx_to_cell.get(ptr >> 1)

    # ── Node allocation (mirrors AllocNode / FreeNode) ──

    def _alloc_node(self) -> int:
        if self._free_head is not None:
            idx = self._free_head
            n = self._nodes[idx]
            self._free_head = n.parent_idx  # reused as free-list next
            n.parent_idx   = None
            n.num_children = 0
            return idx
        self._nodes.append(_BVHNode())
        return len(self._nodes) - 1

    def _free_node(self, node_idx: int) -> None:
        n = self._nodes[node_idx]
        n.parent_idx   = self._free_head   # chain into free list
        n.num_children = 0
        self._free_head = node_idx

    # ── Cost metric: 2-D half-perimeter (mirrors FSurfaceAreaHeuristic) ──

    @staticmethod
    def _cost(aabb: _AABB2) -> float:
        return aabb.cost()

    # ── Greedy best-insertion (mirrors FindBestInsertion_Greedy) ──

    def _find_best_insertion(self, start_node: int, bounds: _AABB2) -> int:
        """
        Greedy descent: at each node pick the child whose merged AABB
        incurs the smallest incremental cost.  Returns the best slot
        (encoded as node_index * _MAX_CHILDREN + child_offset) into which
        the new leaf should be merged.
        """
        best_cost  = float("inf")
        best_slot  = start_node << _INDEX_SHIFT   # default: root slot 0
        node_idx   = start_node
        induced    = 0.0

        while node_idx is not None:
            node = self._nodes[node_idx]
            if not node.is_full():
                # Room to add child without splitting — mirrors non-full branch
                return node_idx << _INDEX_SHIFT   # caller uses parent index
            # Pick spatially closest child (mirrors L1 delta heuristic)
            bx = (bounds.min_x + bounds.max_x) * 0.5
            by = (bounds.min_y + bounds.max_y) * 0.5
            best_child_dist = float("inf")
            best_child_i    = 0
            for i in range(node.num_children):
                cb = node.child_bounds[i]
                cx = (cb.min_x + cb.max_x) * 0.5
                cy = (cb.min_y + cb.max_y) * 0.5
                d  = abs(bx - cx) + abs(by - cy)
                if d < best_child_dist:
                    best_child_dist = d
                    best_child_i    = i

            cb          = node.child_bounds[best_child_i]
            direct_cost = self._cost(bounds.union(cb))
            total_cost  = induced + direct_cost
            child_cost  = total_cost - self._cost(cb)

            if child_cost >= best_cost:
                break
            if total_cost < best_cost:
                best_cost = total_cost
                best_slot = (node_idx << _INDEX_SHIFT) | best_child_i

            ptr = node.child_ptr[best_child_i]
            if ptr & 1:
                break   # leaf — can't descend further

            induced  = child_cost
            node_idx = ptr   # descend into child node

            if induced + self._cost(bounds) >= best_cost:
                break

        return best_slot

    # ── Insert (mirrors FDynamicBVH::Insert) ──

    def _insert(self, bounds: _AABB2, leaf_ptr: int) -> int:
        """
        Insert *leaf_ptr* with bounding box *bounds* into the tree.
        Returns the node-slot that now holds the leaf.
        """
        root_node = self._nodes[self._root]

        # Root still has room — direct insert (fast path)
        if not root_node.is_full():
            slot_i = root_node.num_children
            root_node.num_children += 1
            root_node.child_bounds[slot_i] = bounds
            root_node.child_ptr[slot_i]    = leaf_ptr
            if leaf_ptr & 1:
                self._leaves[self._decode_leaf(leaf_ptr)] = \
                    (self._root << _INDEX_SHIFT) | slot_i
            return (self._root << _INDEX_SHIFT) | slot_i

        # Find best slot via greedy descent
        best_encoded = self._find_best_insertion(self._root, bounds)
        best_node_i  = best_encoded >> _INDEX_SHIFT
        best_child_i = best_encoded & _CHILD_MASK
        best_node    = self._nodes[best_node_i]

        existing_ptr = best_node.child_ptr[best_child_i]
        existing_b   = best_node.child_bounds[best_child_i]
        is_leaf      = bool(existing_ptr & 1)

        # Need a new internal node if slot is a leaf or child node is full
        need_new_level = is_leaf or (
            not is_leaf and self._nodes[existing_ptr].is_full()
        )

        if need_new_level:
            new_node_i = self._alloc_node()
            new_node   = self._nodes[new_node_i]
            new_node.parent_idx   = best_node_i
            new_node.num_children = 1
            new_node.child_bounds[0] = existing_b
            new_node.child_ptr[0]    = existing_ptr
            # Fix back-pointer for moved child
            if existing_ptr & 1:
                cid = self._decode_leaf(existing_ptr)
                if cid is not None:
                    self._leaves[cid] = (new_node_i << _INDEX_SHIFT) | 0
            else:
                self._nodes[existing_ptr].parent_idx = new_node_i

            best_node.child_ptr[best_child_i] = new_node_i
            target_node_i = new_node_i
        else:
            target_node_i = existing_ptr

        target_node = self._nodes[target_node_i]
        slot_i = target_node.num_children
        target_node.num_children += 1
        target_node.child_bounds[slot_i] = bounds
        target_node.child_ptr[slot_i]    = leaf_ptr

        # Propagate bounds up the ancestor chain (mirrors PathBounds loop)
        path_bounds = bounds
        path_slot   = (best_node_i << _INDEX_SHIFT) | best_child_i
        while path_slot is not None:
            pni  = path_slot >> _INDEX_SHIFT
            pci  = path_slot & _CHILD_MASK
            pn   = self._nodes[pni]
            new_b = pn.child_bounds[pci].union(path_bounds)
            pn.child_bounds[pci] = new_b
            path_bounds = new_b
            if pn.parent_idx is None:
                break
            # Find which slot in parent points to this node
            par = self._nodes[pn.parent_idx]
            found = None
            for k in range(par.num_children):
                if par.child_ptr[k] == pni:
                    found = (pn.parent_idx << _INDEX_SHIFT) | k
                    break
            path_slot = found

        return (target_node_i << _INDEX_SHIFT) | slot_i

    # ── Extract (mirrors FDynamicBVH::Extract + RemoveAndSwap) ──

    def _extract(self, slot: int) -> None:
        """
        Remove the entry at *slot* (encoded node-slot) from the tree,
        refitting ancestor bounds.  Mirrors Extract() + RemoveAndSwap().
        """
        node_i = slot >> _INDEX_SHIFT
        child_i = slot & _CHILD_MASK
        node = self._nodes[node_i]

        # RemoveAndSwap: fill gap with last child
        last = node.num_children - 1
        node.num_children = last
        if child_i < last:
            node.child_bounds[child_i] = node.child_bounds[last]
            moved_ptr = node.child_ptr[last]
            node.child_ptr[child_i]   = moved_ptr
            # Fix back-pointer for swapped child
            if moved_ptr & 1:
                cid = self._decode_leaf(moved_ptr)
                if cid is not None:
                    self._leaves[cid] = (node_i << _INDEX_SHIFT) | child_i
            else:
                pass  # inner nodes don't store back-ref by slot
        node.child_bounds[last] = None
        node.child_ptr[last]    = 0

        # Refit ancestor bounds (mirrors PathBounds propagation in Extract)
        path_bounds = node.union_bounds()
        par_i = node.parent_idx

        while par_i is not None:
            par = self._nodes[par_i]
            # Find slot in parent that points to node_i
            for k in range(par.num_children):
                if par.child_ptr[k] == node_i:
                    par.child_bounds[k] = path_bounds
                    break
            path_bounds = par.union_bounds()
            node_i = par_i
            par_i  = par.parent_idx

        # Collapse singleton inner node (mirrors "NumChildren == 1" branch)
        node = self._nodes[node_i if node_i is not None else 0]
        # Find actual bottom node that was modified
        bottom_i  = slot >> _INDEX_SHIFT
        bottom    = self._nodes[bottom_i]
        if not bottom.is_root() and bottom.num_children == 1:
            par_i = bottom.parent_idx
            par   = self._nodes[par_i]
            # Find which slot in parent points to bottom_i
            for k in range(par.num_children):
                if par.child_ptr[k] == bottom_i:
                    par.child_bounds[k] = bottom.child_bounds[0]
                    par.child_ptr[k]    = bottom.child_ptr[0]
                    # Fix back-pointer
                    moved_ptr = bottom.child_ptr[0]
                    if moved_ptr & 1:
                        cid = self._decode_leaf(moved_ptr)
                        if cid is not None:
                            self._leaves[cid] = (par_i << _INDEX_SHIFT) | k
                    else:
                        self._nodes[moved_ptr].parent_idx = par_i
                    break
            self._free_node(bottom_i)
        elif bottom.is_root() and bottom.num_children == 0:
            self._free_node(bottom_i)
            self._root = None

    # ── Batch build recursive (mirrors FDynamicBVH::Build stack) ──

    def _build_recursive(
        self,
        items: List[Tuple[str, _AABB2]],
        parent_i: Optional[int],
    ) -> Optional[int]:
        if not items:
            return None

        node_i = self._alloc_node()
        node   = self._nodes[node_i]
        node.parent_idx = parent_i

        if len(items) <= _MAX_CHILDREN:
            # Leaf-level node
            node.num_children = len(items)
            for i, (cid, ab) in enumerate(items):
                leaf_ptr = self._encode_leaf(cid)
                node.child_bounds[i] = ab
                node.child_ptr[i]    = leaf_ptr
                self._leaves[cid]    = (node_i << _INDEX_SHIFT) | i
            return node_i

        # Split range by highest differing Morton-code bit
        # (mirrors FMortonArray::Split greedy subdivision)
        chunks = self._morton_split(items, _MAX_CHILDREN)

        node.num_children = len(chunks)
        for i, chunk in enumerate(chunks):
            if len(chunk) == 1:
                cid, ab    = chunk[0]
                leaf_ptr   = self._encode_leaf(cid)
                node.child_bounds[i] = ab
                node.child_ptr[i]    = leaf_ptr
                self._leaves[cid]    = (node_i << _INDEX_SHIFT) | i
            else:
                child_i = self._build_recursive(chunk, node_i)
                child_bounds = self._nodes[child_i].union_bounds()
                node.child_bounds[i] = child_bounds
                node.child_ptr[i]    = child_i

        return node_i

    @staticmethod
    def _morton_split(
        items: List[Tuple[str, _AABB2]],
        max_chunks: int,
    ) -> List[List[Tuple[str, _AABB2]]]:
        """
        Recursively bisect *items* (already Morton-sorted) until we have
        at most *max_chunks* groups.  Mirrors FMortonArray::Split bisection
        on the highest differing bit of Morton codes.
        """
        def _bisect(seg: List) -> Tuple[List, List]:
            # Compute 2-D Morton codes for normalised centres within segment
            scene = _AABB2()
            for _, ab in seg:
                scene = scene.union(ab)
            sx = scene.max_x - scene.min_x or 1.0
            sy = scene.max_y - scene.min_y or 1.0

            def _code(ab: _AABB2) -> int:
                cx = ((ab.min_x + ab.max_x) * 0.5 - scene.min_x) / sx
                cy = ((ab.min_y + ab.max_y) * 0.5 - scene.min_y) / sy
                xi = int(max(0.0, min(1023.0, cx * 1023.0)))
                yi = int(max(0.0, min(1023.0, cy * 1023.0)))
                def _sp(v: int) -> int:
                    v &= 0x3FF
                    v = (v | (v << 8)) & 0x00FF00FF
                    v = (v | (v << 4)) & 0x0F0F0F0F
                    v = (v | (v << 2)) & 0x33333333
                    v = (v | (v << 1)) & 0x55555555
                    return v
                return _sp(xi) | (_sp(yi) << 1)

            codes = [_code(ab) for _, ab in seg]
            diff  = codes[0] ^ codes[-1]
            if diff == 0:
                mid = len(seg) // 2
            else:
                hb   = int(_math.floor(_math.log2(diff))) if diff else 0
                mask = 1 << hb
                mid  = len(seg) // 2   # fallback
                for k in range(len(seg)):
                    if codes[k] & mask:
                        mid = k
                        break
                if mid == 0:
                    mid = len(seg) // 2
            return seg[:mid], seg[mid:]

        chunks = [items]
        while len(chunks) < max_chunks:
            # Pick the largest chunk to split
            largest_i = max(range(len(chunks)), key=lambda i: len(chunks[i]))
            if len(chunks[largest_i]) <= 1:
                break
            left, right = _bisect(chunks[largest_i])
            if not left or not right:
                break
            chunks[largest_i] = left
            chunks.append(right)

        return [c for c in chunks if c]


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] ShadingEnergyConservation → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/ShadingEnergyConservation.cpp
#
# Core physics: total radiant energy leaving a cell surface must not exceed
# the energy arriving.  In the SVG substrate the three light-transport channels
# are fill opacity (diffuse bulk), stroke opacity (specular edge), and shadow
# opacity (sub-surface / transmission leak).  Their weighted sum must satisfy:
#
#   E_fill * w_fill + E_stroke * w_stroke + E_shadow * w_shadow <= 1.0
#
# which mirrors the furnace-test invariant from the C++ implementation:
#   integral of BRDF * cos(theta) dOmega <= 1.0
#
# LUT textures (GGXSpecEnergy, ClothEnergy, DiffuseEnergy, …) are replaced by
# analytic approximations (鲁迅式: no texture lookup, just physics constraints):
#
#   GGX specular directional albedo  → Schlick + Smith analytic fit
#   Cloth (Ashikhmin sheen)          → sinusoidal edge rolloff
#   Multi-scattering compensation    → geometric series sum (1/(1-E_avg))
#
# Energy conservation flag (r.Shading.EnergyConservation = 1 → always on):
#   Clamp weighted-channel sum to 1.0, distributing excess proportionally.
#
# Energy preservation flag (r.Shading.EnergyPreservation = 1 → always on):
#   Attenuate fill (diffuse) channel by specular directional albedo so specular
#   reflection correctly steals energy from diffuse — same as the C++ path that
#   multiplies DiffuseColor by (1 - SpecularEnergy).
#
# Species-to-roughness mapping reuses _SPECIES_ROUGHNESS from the ReflectionEnv
# port; roughness drives the GGX lobe width (alpha = roughness^2).
# ═══════════════════════════════════════════════════════════════════════════════

# Channel weights for the three SVG light-transport bins.
# Mirrors the per-lobe energy split in the substrate BRDF evaluation:
#   diffuse (fill)     → bulk transmission through the cell body
#   specular (stroke)  → edge Fresnel reflection
#   shadow             → energy lost to sub-surface / transmission
_SEC_WEIGHT_FILL:   float = 0.60   # GGX diffuse lobe budget (Lambertian approx)
_SEC_WEIGHT_STROKE: float = 0.28   # GGX specular lobe budget
_SEC_WEIGHT_SHADOW: float = 0.12   # transmission / shadow lobe budget

# Minimum channel opacity floor — even at maximum energy loss the channel
# must remain visible (mirrors the C++ fallback BlackDummy texture path).
_SEC_OPACITY_FLOOR: float = 0.05

# Cosine of the representative view angle used for directional-albedo queries.
# 45° (cos = 1/√2) is the representative half-angle used in the analytic fits
# when no per-pixel angle is available — same as the C++ furnace-test default.
_SEC_COS_VIEW: float = 0.7071067811865476   # cos(pi/4)




