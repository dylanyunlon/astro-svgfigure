// BVHTree.ts — Dynamic BVH for broad-phase collision detection
// Modeled after Box2D b2DynamicTree

import { AABB, aabbOverlap, aabbUnion, aabbArea, aabbExpand, aabbContains } from './AABB';

const NULL_NODE = -1;
const FAT_MARGIN = 0.05; // AABB expansion for temporal coherence

interface BVHNode {
  aabb: AABB;
  parent: number;
  left: number;
  right: number;
  height: number;
  userData: number; // body index for leaves, -1 for internal
  moved: boolean;
}

export class BVHTree {
  private nodes: BVHNode[];
  private root: number = NULL_NODE;
  private nodeCount: number = 0;
  private nodeCapacity: number;
  private freeList: number = 0;

  constructor(initialCapacity = 256) {
    this.nodeCapacity = initialCapacity;
    this.nodes = new Array(initialCapacity);
    for (let i = 0; i < initialCapacity - 1; i++) {
      this.nodes[i] = this.createNode();
      this.nodes[i].parent = i + 1; // free list chain
      this.nodes[i].height = -1;
    }
    this.nodes[initialCapacity - 1] = this.createNode();
    this.nodes[initialCapacity - 1].parent = NULL_NODE;
    this.nodes[initialCapacity - 1].height = -1;
  }

  private createNode(): BVHNode {
    return {
      aabb: { minX: 0, minY: 0, maxX: 0, maxY: 0 },
      parent: NULL_NODE, left: NULL_NODE, right: NULL_NODE,
      height: -1, userData: -1, moved: false,
    };
  }

  private allocateNode(): number {
    if (this.freeList === NULL_NODE) {
      // Grow pool
      const old = this.nodeCapacity;
      this.nodeCapacity *= 2;
      for (let i = old; i < this.nodeCapacity - 1; i++) {
        this.nodes[i] = this.createNode();
        this.nodes[i].parent = i + 1;
      }
      this.nodes[this.nodeCapacity - 1] = this.createNode();
      this.nodes[this.nodeCapacity - 1].parent = NULL_NODE;
      this.freeList = old;
    }
    const id = this.freeList;
    this.freeList = this.nodes[id].parent;
    this.nodes[id].parent = NULL_NODE;
    this.nodes[id].left = NULL_NODE;
    this.nodes[id].right = NULL_NODE;
    this.nodes[id].height = 0;
    this.nodes[id].userData = -1;
    this.nodes[id].moved = false;
    this.nodeCount++;
    return id;
  }

  private freeNode(id: number): void {
    this.nodes[id].parent = this.freeList;
    this.nodes[id].height = -1;
    this.nodes[id].userData = -1;
    this.freeList = id;
    this.nodeCount--;
  }

  private isLeaf(id: number): boolean {
    return this.nodes[id].left === NULL_NODE;
  }

  insert(bodyId: number, aabb: AABB): number {
    const proxyId = this.allocateNode();
    const node = this.nodes[proxyId];
    node.aabb = aabbExpand(aabb, FAT_MARGIN);
    node.userData = bodyId;
    node.height = 0;
    this.insertLeaf(proxyId);
    return proxyId;
  }

  remove(proxyId: number): void {
    this.removeLeaf(proxyId);
    this.freeNode(proxyId);
  }

  moveProxy(proxyId: number, newAABB: AABB, displacement: { x: number; y: number }): boolean {
    const node = this.nodes[proxyId];
    // If still inside fat AABB, no work needed (temporal coherence)
    if (aabbContains(node.aabb, newAABB)) {
      return false;
    }
    this.removeLeaf(proxyId);
    // Expand by margin + displacement prediction
    let expanded = aabbExpand(newAABB, FAT_MARGIN);
    const dx = displacement.x * 2.0;
    const dy = displacement.y * 2.0;
    if (dx < 0) expanded.minX += dx; else expanded.maxX += dx;
    if (dy < 0) expanded.minY += dy; else expanded.maxY += dy;
    node.aabb = expanded;
    this.insertLeaf(proxyId);
    node.moved = true;
    return true;
  }

  query(aabb: AABB, callback: (proxyId: number) => boolean): void {
    const stack: number[] = [];
    if (this.root !== NULL_NODE) stack.push(this.root);
    while (stack.length > 0) {
      const nodeId = stack.pop()!;
      if (nodeId === NULL_NODE) continue;
      const node = this.nodes[nodeId];
      if (!aabbOverlap(node.aabb, aabb)) continue;
      if (this.isLeaf(nodeId)) {
        const proceed = callback(nodeId);
        if (!proceed) return;
      } else {
        stack.push(node.left);
        stack.push(node.right);
      }
    }
  }

  getUserData(proxyId: number): number {
    return this.nodes[proxyId].userData;
  }

  getFatAABB(proxyId: number): AABB {
    return this.nodes[proxyId].aabb;
  }

  // ── Internal: insertion with SAH ──────────────────────────────────────
  private insertLeaf(leaf: number): void {
    if (this.root === NULL_NODE) {
      this.root = leaf;
      this.nodes[leaf].parent = NULL_NODE;
      return;
    }
    const leafAABB = this.nodes[leaf].aabb;
    let index = this.root;

    while (!this.isLeaf(index)) {
      const node = this.nodes[index];
      const combinedArea = aabbArea(aabbUnion(node.aabb, leafAABB));
      const cost = 2.0 * combinedArea;
      const inheritanceCost = 2.0 * (combinedArea - aabbArea(node.aabb));

      // Cost of descending into left
      const left = node.left;
      let costLeft: number;
      if (this.isLeaf(left)) {
        costLeft = aabbArea(aabbUnion(this.nodes[left].aabb, leafAABB)) + inheritanceCost;
      } else {
        const oldArea = aabbArea(this.nodes[left].aabb);
        const newArea = aabbArea(aabbUnion(this.nodes[left].aabb, leafAABB));
        costLeft = (newArea - oldArea) + inheritanceCost;
      }

      // Cost of descending into right
      const right = node.right;
      let costRight: number;
      if (this.isLeaf(right)) {
        costRight = aabbArea(aabbUnion(this.nodes[right].aabb, leafAABB)) + inheritanceCost;
      } else {
        const oldArea = aabbArea(this.nodes[right].aabb);
        const newArea = aabbArea(aabbUnion(this.nodes[right].aabb, leafAABB));
        costRight = (newArea - oldArea) + inheritanceCost;
      }

      // SAH: create new parent here if cheapest
      if (cost < costLeft && cost < costRight) break;
      index = costLeft < costRight ? left : right;
    }

    const sibling = index;
    const oldParent = this.nodes[sibling].parent;
    const newParent = this.allocateNode();
    this.nodes[newParent].parent = oldParent;
    this.nodes[newParent].aabb = aabbUnion(leafAABB, this.nodes[sibling].aabb);
    this.nodes[newParent].height = this.nodes[sibling].height + 1;
    this.nodes[newParent].left = sibling;
    this.nodes[newParent].right = leaf;
    this.nodes[sibling].parent = newParent;
    this.nodes[leaf].parent = newParent;

    if (oldParent !== NULL_NODE) {
      if (this.nodes[oldParent].left === sibling) {
        this.nodes[oldParent].left = newParent;
      } else {
        this.nodes[oldParent].right = newParent;
      }
    } else {
      this.root = newParent;
    }

    // Walk back up fixing heights and AABBs
    let n = this.nodes[leaf].parent;
    while (n !== NULL_NODE) {
      n = this.balance(n);
      const left = this.nodes[n].left;
      const right = this.nodes[n].right;
      this.nodes[n].height = 1 + Math.max(this.nodes[left].height, this.nodes[right].height);
      this.nodes[n].aabb = aabbUnion(this.nodes[left].aabb, this.nodes[right].aabb);
      n = this.nodes[n].parent;
    }
  }

  private removeLeaf(leaf: number): void {
    if (leaf === this.root) { this.root = NULL_NODE; return; }
    const parent = this.nodes[leaf].parent;
    const grandParent = this.nodes[parent].parent;
    const sibling = this.nodes[parent].left === leaf ? this.nodes[parent].right : this.nodes[parent].left;

    if (grandParent !== NULL_NODE) {
      if (this.nodes[grandParent].left === parent) {
        this.nodes[grandParent].left = sibling;
      } else {
        this.nodes[grandParent].right = sibling;
      }
      this.nodes[sibling].parent = grandParent;
      this.freeNode(parent);

      let n = grandParent;
      while (n !== NULL_NODE) {
        n = this.balance(n);
        const l = this.nodes[n].left, r = this.nodes[n].right;
        this.nodes[n].aabb = aabbUnion(this.nodes[l].aabb, this.nodes[r].aabb);
        this.nodes[n].height = 1 + Math.max(this.nodes[l].height, this.nodes[r].height);
        n = this.nodes[n].parent;
      }
    } else {
      this.root = sibling;
      this.nodes[sibling].parent = NULL_NODE;
      this.freeNode(parent);
    }
  }

  private balance(a: number): number {
    if (this.isLeaf(a) || this.nodes[a].height < 2) return a;
    const b = this.nodes[a].left;
    const c = this.nodes[a].right;
    const balance = this.nodes[c].height - this.nodes[b].height;
    // Rotate C up
    if (balance > 1) return this.rotate(a, c, b, true);
    // Rotate B up
    if (balance < -1) return this.rotate(a, b, c, false);
    return a;
  }

  private rotate(a: number, high: number, low: number, highIsRight: boolean): number {
    const hL = this.nodes[high].left, hR = this.nodes[high].right;
    // high becomes parent of a
    this.nodes[high].parent = this.nodes[a].parent;
    if (this.nodes[a].parent !== NULL_NODE) {
      if (this.nodes[this.nodes[a].parent].left === a) this.nodes[this.nodes[a].parent].left = high;
      else this.nodes[this.nodes[a].parent].right = high;
    } else {
      this.root = high;
    }
    // Decide which child of high goes down
    if (this.nodes[hL].height > this.nodes[hR].height) {
      this.nodes[high].right = a; // hR goes to a
      this.nodes[a].parent = high;
      if (highIsRight) this.nodes[a].left = low; // keep low
      else this.nodes[a].right = low;
      // Actually: simpler standard AVL rotation
      this.nodes[hR].parent = a;
      if (highIsRight) this.nodes[a].right = hR;
      else this.nodes[a].left = hR;
    } else {
      this.nodes[high].left = a;
      this.nodes[a].parent = high;
      this.nodes[hL].parent = a;
      if (highIsRight) this.nodes[a].right = hL;
      else this.nodes[a].left = hL;
    }
    this.nodes[a].aabb = aabbUnion(this.nodes[this.nodes[a].left].aabb, this.nodes[this.nodes[a].right].aabb);
    this.nodes[a].height = 1 + Math.max(this.nodes[this.nodes[a].left].height, this.nodes[this.nodes[a].right].height);
    this.nodes[high].aabb = aabbUnion(this.nodes[this.nodes[high].left].aabb, this.nodes[this.nodes[high].right].aabb);
    this.nodes[high].height = 1 + Math.max(this.nodes[this.nodes[high].left].height, this.nodes[this.nodes[high].right].height);
    return high;
  }
}
