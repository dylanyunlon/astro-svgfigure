#version 300 es
/**
 * constraint-shader.frag — GPGPU cell constraint update pass
 *
 * Active Theory Neon 风格：把 physics_engine 的 CPU 碰撞约束搬到 GPU。
 * 每帧对每个像素（= 每个 cell）并行执行。
 *
 * 输入纹理 (uState) RGBA 编码：
 *   R = dx          encode([-DISP_MAX, DISP_MAX] → [0,1])
 *   G = dy          encode([-DISP_MAX, DISP_MAX] → [0,1])
 *   B = force_mag   encode([0, FORCE_MAX] → [0,1])
 *   A = converged   0.0 / 1.0
 *
 * 输出：更新后的同格式 RGBA。
 *
 * 邻居碰撞力计算：
 *   对上下左右四个相邻像素（= 相邻 cells）采样。
 *   若两个 cell 重叠（通过 dx/dy 推断），产生排斥力。
 *   力的方向 = 从邻居中心指向当前 cell 中心。
 *   力的大小 = max(0, CELL_RADIUS * 2 - dist) * SPRING_K，超出则裁剪。
 *   四邻居力叠加后更新 dx, dy；同时更新 force_mag 和 converged flag。
 *
 * 参考: channels/physics/force_field.json, collision.json
 */

precision highp float;

// ── Uniforms ─────────────────────────────────────────────────────────────────

/** 当前约束状态纹理（上一帧输出） */
uniform sampler2D uState;

/** 纹理尺寸 (texSize x texSize) */
uniform float uTexSize;

/** 每个 cell 的半径（像素单位，用于碰撞检测） */
uniform float uCellRadius;      // default ~40.0

/** 弹簧刚度系数 */
uniform float uSpringK;         // default ~0.15

/** 阻尼系数（速度衰减） */
uniform float uDamping;         // default ~0.85

/** 时间步长 (dt in frames, usually 1.0) */
uniform float uDt;

/** 仅在该帧强制所有像素更新（epoch reset） */
uniform float uForceUpdate;     // 0.0 / 1.0

// ── Varyings ──────────────────────────────────────────────────────────────────

in vec2 vUV;
out vec4 fragColor;

// ── Constants ─────────────────────────────────────────────────────────────────

const float DISP_MAX  = 500.0;
const float FORCE_MAX = 200.0;

/** 收敛阈值：force_mag < 此值时标记 converged */
const float CONVERGE_THRESH = 0.5;   // px

/** 最大邻居排斥力（防止爆炸） */
const float MAX_REPULSION = 80.0;

// ── Encode / Decode helpers ───────────────────────────────────────────────────

float encodeDisp(float v) {
  return clamp((v + DISP_MAX) / (2.0 * DISP_MAX), 0.0, 1.0);
}

float decodeDisp(float v) {
  return v * 2.0 * DISP_MAX - DISP_MAX;
}

float encodeForce(float v) {
  return clamp(v / FORCE_MAX, 0.0, 1.0);
}

// ── Sample a neighbour at texel offset (di, dj) ───────────────────────────────
// Returns (decoded_dx, decoded_dy, decoded_force, converged) as vec4.
vec4 sampleNeighbour(vec2 pixelCoord, float di, float dj) {
  vec2 neighbUV = (pixelCoord + vec2(di, dj) + 0.5) / uTexSize;
  // Clamp to texture border — out-of-range = no cell exists there
  neighbUV = clamp(neighbUV, 0.0, 1.0);
  vec4 raw = texture(uState, neighbUV);
  return vec4(
    decodeDisp(raw.r),   // dx
    decodeDisp(raw.g),   // dy
    raw.b * FORCE_MAX,   // force_mag
    raw.a                // converged flag
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {
  // Current pixel in texel coordinates
  vec2 pixelCoord = floor(vUV * uTexSize);

  // Sample own state
  vec4 self   = texture(uState, vUV);
  float selfDx    = decodeDisp(self.r);
  float selfDy    = decodeDisp(self.g);
  float selfForce = self.b * FORCE_MAX;
  float selfConv  = self.a;

  // ── Short-circuit if already converged (unless force update) ──────────────
  if (selfConv > 0.5 && uForceUpdate < 0.5) {
    fragColor = self;
    return;
  }

  // ── Accumulate neighbour repulsion forces ────────────────────────────────
  // Neighbours: left(-1,0), right(+1,0), up(0,-1), down(0,+1)
  // (in texel space; each texel = one cell)
  vec2 dirs[4];
  dirs[0] = vec2(-1.0,  0.0);
  dirs[1] = vec2( 1.0,  0.0);
  dirs[2] = vec2( 0.0, -1.0);
  dirs[3] = vec2( 0.0,  1.0);

  float accX = 0.0;
  float accY = 0.0;

  for (int k = 0; k < 4; k++) {
    vec2  dir   = dirs[k];
    vec4  nb    = sampleNeighbour(pixelCoord, dir.x, dir.y);
    float nbDx  = nb.x;
    float nbDy  = nb.y;

    // World-space separation vector between this cell and neighbour:
    // In the layout, each cell has its own offset (dx, dy) from its rest pos.
    // We approximate overlap by comparing offsets along the direction.
    // True spatial distance would require a separate position texture;
    // here we use a simpler proxy: offset delta in the direction of separation.
    float sepX = selfDx - nbDx;
    float sepY = selfDy - nbDy;

    // Project separation onto neighbour direction (1 texel ≈ 1 cell diameter)
    // The minimum separation distance = 2 * radius
    float minDist = uCellRadius * 2.0;

    // Distance between cell centers in displacement-space
    float dist = sqrt(sepX * sepX + sepY * sepY) + 0.001;

    // Repulsion only when closer than minDist
    float overlap = max(0.0, minDist - dist);

    if (overlap > 0.0) {
      // Push along separation direction
      float repulse = min(overlap * uSpringK, MAX_REPULSION);
      // Normalise separation vector
      accX += (sepX / dist) * repulse;
      accY += (sepY / dist) * repulse;
    }
  }

  // ── Integrate: update displacement ───────────────────────────────────────
  // dx_new = dx_old * damping + accumulated_force * dt
  float newDx = selfDx * uDamping + accX * uDt;
  float newDy = selfDy * uDamping + accY * uDt;

  // Total force magnitude
  float newForce = sqrt(accX * accX + accY * accY);

  // Converged when net force is negligible
  float newConv = (newForce < CONVERGE_THRESH) ? 1.0 : 0.0;

  // ── Encode and output ─────────────────────────────────────────────────────
  fragColor = vec4(
    encodeDisp(newDx),
    encodeDisp(newDy),
    encodeForce(newForce),
    newConv
  );
}
