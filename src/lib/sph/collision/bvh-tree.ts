import { AABB } from './aabb-manager';

// ─── Node ────────────────────────────────────────────────────────────────────

export interface BVHNode {
  aabb: AABB;
  parent: number;   // -1 = none
  left: number;     // -1 = none
  right: number;    // -1 = none
  height: number;
  bodyId: number;   // -1 = internal node
}

const NULL = -1;
const FATTEN = 2;

function fattenAABB(a: AABB): AABB {
  return {
    minX: a.minX - FATTEN,
    minY: a.minY - FATTEN,
    maxX: a.maxX + FATTEN,
    maxY: a.maxY + FATTEN,
  };
}

function unionAABB(a: AABB, b: AABB): AABB {
  return {
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
  };
}

function aabbArea(a: AABB): number {
  return (a.maxX - a.minX) * (a.maxY - a.minY);
}

function aabbPerimeter(a: AABB): number {
  return 2 * ((a.maxX - a.minX) + (a.maxY - a.minY));
}

function containsAABB(outer: AABB, inner: AABB): boolean {
  return (
    outer.minX <= inner.minX &&
    outer.minY <= inner.minY &&
    outer.maxX >= inner.maxX &&
    outer.maxY >= inner.maxY
  );
}

function overlapsAABB(a: AABB, b: AABB): boolean {
  return (
    a.minX <= b.maxX && a.maxX >= b.minX &&
    a.minY <= b.maxY && a.maxY >= b.minY
  );
}

export interface RaycastHit {
  bodyId: number;
  t: number;
}

// ─── BVHTree ─────────────────────────────────────────────────────────────────

export class BVHTree {
  private nodes: BVHNode[] = [];
  private freeList: number[] = [];
  private root: number = NULL;

  // proxy map: bodyId → leaf nodeIndex
  private proxyMap = new Map<number, number>();

  // ── allocate / free ────────────────────────────────────────────────────────

  private allocNode(): number {
    if (this.freeList.length > 0) {
      const idx = this.freeList.pop()!;
      this.nodes[idx] = {
        aabb: { minX: 0, minY: 0, maxX: 0, maxY: 0 },
        parent: NULL, left: NULL, right: NULL,
        height: 0, bodyId: NULL,
      };
      return idx;
    }
    const idx = this.nodes.length;
    this.nodes.push({
      aabb: { minX: 0, minY: 0, maxX: 0, maxY: 0 },
      parent: NULL, left: NULL, right: NULL,
      height: 0, bodyId: NULL,
    });
    return idx;
  }

  private freeNode(idx: number): void {
    this.freeList.push(idx);
  }

  // ── insert ─────────────────────────────────────────────────────────────────

  insert(bodyId: number, aabb: AABB): void {
    const leaf = this.allocNode();
    const n = this.nodes[leaf];
    n.aabb = fattenAABB(aabb);
    n.bodyId = bodyId;
    n.height = 0;
    this.proxyMap.set(bodyId, leaf);
    this.insertLeaf(leaf);
  }

  private insertLeaf(leaf: number): void {
    if (this.root === NULL) {
      this.root = leaf;
      this.nodes[this.root].parent = NULL;
      return;
    }

    // SAH best-sibling search
    const leafAABB = this.nodes[leaf].aabb;
    let best = this.root;
    let bestCost = aabbPerimeter(unionAABB(leafAABB, this.nodes[this.root].aabb));

    // priority queue entries: [nodeIndex, inheritedCost]
    const stack: Array<[number, number]> = [[this.root, 0]];

    while (stack.length > 0) {
      const [idx, inherited] = stack.pop()!;
      const node = this.nodes[idx];
      const directCost = aabbPerimeter(unionAABB(leafAABB, node.aabb));
      const totalCost = directCost + inherited;

      if (totalCost < bestCost) {
        bestCost = totalCost;
        best = idx;
      }

      if (node.left !== NULL || node.right !== NULL) {
        // lower bound for children
        const childInherited = inherited + (directCost - aabbPerimeter(node.aabb));
        const lowerBound = aabbPerimeter(leafAABB) + childInherited;
        if (lowerBound < bestCost) {
          if (node.left !== NULL)  stack.push([node.left,  childInherited]);
          if (node.right !== NULL) stack.push([node.right, childInherited]);
        }
      }
    }

    // create new parent
    const oldParent = this.nodes[best].parent;
    const newParent = this.allocNode();
    const np = this.nodes[newParent];
    np.aabb = unionAABB(leafAABB, this.nodes[best].aabb);
    np.parent = oldParent;
    np.height = this.nodes[best].height + 1;
    np.bodyId = NULL;

    if (oldParent !== NULL) {
      if (this.nodes[oldParent].left === best) {
        this.nodes[oldParent].left = newParent;
      } else {
        this.nodes[oldParent].right = newParent;
      }
    } else {
      this.root = newParent;
    }

    np.left = best;
    np.right = leaf;
    this.nodes[best].parent = newParent;
    this.nodes[leaf].parent = newParent;

    this.fixUpwards(newParent);
  }

  // ── remove ─────────────────────────────────────────────────────────────────

  remove(bodyId: number): void {
    const leaf = this.proxyMap.get(bodyId);
    if (leaf === undefined) return;
    this.proxyMap.delete(bodyId);
    this.removeLeaf(leaf);
    this.freeNode(leaf);
  }

  private removeLeaf(leaf: number): void {
    if (leaf === this.root) {
      this.root = NULL;
      return;
    }

    const parent = this.nodes[leaf].parent;
    const grandParent = this.nodes[parent].parent;
    const sibling = this.nodes[parent].left === leaf
      ? this.nodes[parent].right
      : this.nodes[parent].left;

    if (grandParent !== NULL) {
      if (this.nodes[grandParent].left === parent) {
        this.nodes[grandParent].left = sibling;
      } else {
        this.nodes[grandParent].right = sibling;
      }
      if (sibling !== NULL) this.nodes[sibling].parent = grandParent;
      this.freeNode(parent);
      this.fixUpwards(grandParent);
    } else {
      this.root = sibling;
      if (sibling !== NULL) this.nodes[sibling].parent = NULL;
      this.freeNode(parent);
    }
  }

  // ── moveProxy ──────────────────────────────────────────────────────────────

  moveProxy(bodyId: number, aabb: AABB): boolean {
    const leaf = this.proxyMap.get(bodyId);
    if (leaf === undefined) return false;
    const fat = fattenAABB(aabb);
    if (containsAABB(this.nodes[leaf].aabb, fat)) return false;
    this.removeLeaf(leaf);
    this.nodes[leaf].aabb = fat;
    this.insertLeaf(leaf);
    return true;
  }

  // ── fix heights & AABBs upwards, then AVL rebalance ───────────────────────

  private fixUpwards(idx: number): void {
    let cur = idx;
    while (cur !== NULL) {
      cur = this.balance(cur);
      const n = this.nodes[cur];
      const lh = n.left  !== NULL ? this.nodes[n.left].height  : -1;
      const rh = n.right !== NULL ? this.nodes[n.right].height : -1;
      n.height = 1 + Math.max(lh, rh);
      if (n.left !== NULL && n.right !== NULL) {
        n.aabb = unionAABB(this.nodes[n.left].aabb, this.nodes[n.right].aabb);
      }
      cur = n.parent;
    }
  }

  // ── AVL rotation ──────────────────────────────────────────────────────────

  private balance(idx: number): number {
    const A = this.nodes[idx];
    if (A.left === NULL || A.right === NULL || A.height < 2) return idx;

    const lh = this.nodes[A.left].height;
    const rh = this.nodes[A.right].height;
    const balance = lh - rh;

    // right-heavy → rotate left
    if (balance < -1) {
      const B = A.right;
      return this.rotateLeft(idx, B);
    }
    // left-heavy → rotate right
    if (balance > 1) {
      const B = A.left;
      return this.rotateRight(idx, B);
    }
    return idx;
  }

  private rotateLeft(idxA: number, idxB: number): number {
    const A = this.nodes[idxA];
    const B = this.nodes[idxB];
    const Bleft  = B.left;
    const Bright = B.right;

    // B takes A's position
    B.left   = idxA;
    B.parent = A.parent;
    A.parent = idxB;

    if (B.parent !== NULL) {
      if (this.nodes[B.parent].left === idxA) {
        this.nodes[B.parent].left = idxB;
      } else {
        this.nodes[B.parent].right = idxB;
      }
    } else {
      this.root = idxB;
    }

    // pick which of B's children to hand to A
    const Blh = Bleft  !== NULL ? this.nodes[Bleft].height  : -1;
    const Brh = Bright !== NULL ? this.nodes[Bright].height : -1;
    if (Blh > Brh) {
      B.right  = Bleft!;
      A.right  = Bright !== NULL ? Bright : NULL;
      if (A.right !== NULL) this.nodes[A.right].parent = idxA;
      if (B.right !== NULL) this.nodes[B.right].parent = idxB;
    } else {
      B.right  = Bright !== NULL ? Bright : NULL;
      A.right  = Bleft !== NULL ? Bleft : NULL;
      if (A.right !== NULL) this.nodes[A.right].parent = idxA;
      if (B.right !== NULL) this.nodes[B.right].parent = idxB;
    }

    // recompute heights and AABBs
    if (A.left !== NULL && A.right !== NULL)
      A.aabb = unionAABB(this.nodes[A.left].aabb, this.nodes[A.right].aabb);
    A.height = 1 + Math.max(
      A.left  !== NULL ? this.nodes[A.left].height  : -1,
      A.right !== NULL ? this.nodes[A.right].height : -1,
    );
    if (B.left !== NULL && B.right !== NULL)
      B.aabb = unionAABB(this.nodes[B.left].aabb, this.nodes[B.right].aabb);
    B.height = 1 + Math.max(
      B.left  !== NULL ? this.nodes[B.left].height  : -1,
      B.right !== NULL ? this.nodes[B.right].height : -1,
    );
    return idxB;
  }

  private rotateRight(idxA: number, idxB: number): number {
    const A = this.nodes[idxA];
    const B = this.nodes[idxB];
    const Bleft  = B.left;
    const Bright = B.right;

    B.right  = idxA;
    B.parent = A.parent;
    A.parent = idxB;

    if (B.parent !== NULL) {
      if (this.nodes[B.parent].left === idxA) {
        this.nodes[B.parent].left = idxB;
      } else {
        this.nodes[B.parent].right = idxB;
      }
    } else {
      this.root = idxB;
    }

    const Blh = Bleft  !== NULL ? this.nodes[Bleft].height  : -1;
    const Brh = Bright !== NULL ? this.nodes[Bright].height : -1;
    if (Blh > Brh) {
      B.left  = Bleft !== NULL ? Bleft : NULL;
      A.left  = Bright !== NULL ? Bright : NULL;
      if (A.left !== NULL) this.nodes[A.left].parent = idxA;
      if (B.left !== NULL) this.nodes[B.left].parent = idxB;
    } else {
      B.left  = Bright !== NULL ? Bright : NULL;
      A.left  = Bleft !== NULL ? Bleft : NULL;
      if (A.left !== NULL) this.nodes[A.left].parent = idxA;
      if (B.left !== NULL) this.nodes[B.left].parent = idxB;
    }

    if (A.left !== NULL && A.right !== NULL)
      A.aabb = unionAABB(this.nodes[A.left].aabb, this.nodes[A.right].aabb);
    A.height = 1 + Math.max(
      A.left  !== NULL ? this.nodes[A.left].height  : -1,
      A.right !== NULL ? this.nodes[A.right].height : -1,
    );
    if (B.left !== NULL && B.right !== NULL)
      B.aabb = unionAABB(this.nodes[B.left].aabb, this.nodes[B.right].aabb);
    B.height = 1 + Math.max(
      B.left  !== NULL ? this.nodes[B.left].height  : -1,
      B.right !== NULL ? this.nodes[B.right].height : -1,
    );
    return idxB;
  }

  // ── queryAABB ──────────────────────────────────────────────────────────────

  queryAABB(aabb: AABB): number[] {
    const result: number[] = [];
    if (this.root === NULL) return result;
    const stack: number[] = [this.root];
    while (stack.length > 0) {
      const idx = stack.pop()!;
      const n = this.nodes[idx];
      if (!overlapsAABB(n.aabb, aabb)) continue;
      if (n.bodyId !== NULL) {
        result.push(n.bodyId);
      } else {
        if (n.left  !== NULL) stack.push(n.left);
        if (n.right !== NULL) stack.push(n.right);
      }
    }
    return result;
  }

  // ── queryAllPairs ──────────────────────────────────────────────────────────

  queryAllPairs(): Array<[number, number]> {
    const pairs: Array<[number, number]> = [];
    const leaves: number[] = [];

    // collect all leaves
    if (this.root === NULL) return pairs;
    const stack: number[] = [this.root];
    while (stack.length > 0) {
      const idx = stack.pop()!;
      const n = this.nodes[idx];
      if (n.bodyId !== NULL) {
        leaves.push(idx);
      } else {
        if (n.left  !== NULL) stack.push(n.left);
        if (n.right !== NULL) stack.push(n.right);
      }
    }

    for (let i = 0; i < leaves.length; i++) {
      const a = this.nodes[leaves[i]];
      for (let j = i + 1; j < leaves.length; j++) {
        const b = this.nodes[leaves[j]];
        if (overlapsAABB(a.aabb, b.aabb)) {
          pairs.push([a.bodyId, b.bodyId]);
        }
      }
    }
    return pairs;
  }

  // ── raycast ───────────────────────────────────────────────────────────────

  raycast(ox: number, oy: number, dx: number, dy: number, maxT = Infinity): RaycastHit[] {
    const hits: RaycastHit[] = [];
    if (this.root === NULL) return hits;

    const invDx = dx !== 0 ? 1 / dx : Infinity;
    const invDy = dy !== 0 ? 1 / dy : Infinity;

    const stack: number[] = [this.root];
    while (stack.length > 0) {
      const idx = stack.pop()!;
      const n = this.nodes[idx];

      const t = this.rayAABB(ox, oy, invDx, invDy, n.aabb);
      if (t === null || t > maxT) continue;

      if (n.bodyId !== NULL) {
        hits.push({ bodyId: n.bodyId, t });
      } else {
        if (n.left  !== NULL) stack.push(n.left);
        if (n.right !== NULL) stack.push(n.right);
      }
    }

    hits.sort((a, b) => a.t - b.t);
    return hits;
  }

  private rayAABB(
    ox: number, oy: number,
    invDx: number, invDy: number,
    box: AABB,
  ): number | null {
    const tx1 = (box.minX - ox) * invDx;
    const tx2 = (box.maxX - ox) * invDx;
    const ty1 = (box.minY - oy) * invDy;
    const ty2 = (box.maxY - oy) * invDy;

    const tmin = Math.max(Math.min(tx1, tx2), Math.min(ty1, ty2));
    const tmax = Math.min(Math.max(tx1, tx2), Math.max(ty1, ty2));

    if (tmax < 0 || tmin > tmax) return null;
    return tmin >= 0 ? tmin : tmax;
  }

  // ── debug helpers ─────────────────────────────────────────────────────────

  getHeight(): number {
    return this.root === NULL ? 0 : this.nodes[this.root].height;
  }

  getNodeCount(): number {
    return this.nodes.length - this.freeList.length;
  }
}
