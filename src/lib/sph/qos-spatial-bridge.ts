// src/lib/sph/qos-spatial-bridge.ts
// QoS → Physics mapping for SPH spatial renderer
// Mirrors qos_spatial.py (task 07) — same formulas, TypeScript types

export type Reliability = 'RELIABLE' | 'BEST_EFFORT';
export type Durability  = 'VOLATILE' | 'TRANSIENT_LOCAL';

export interface QoSProfile {
  reliability:  Reliability;
  durability:   Durability;
  historyDepth: number;
  mps:          number;
  priority:     number; // 0-3
}

export interface SpatialPhysics {
  viscosity:        number;
  boundaryFriction: number;
  trailLength:      number;
  emitterRate:      number;
  forceMultiplier:  number;
}

// ---------------------------------------------------------------------------
// Core mapping function — identical formulas to Python qos_to_physics()
// ---------------------------------------------------------------------------
export function qosToSpatial(qos: QoSProfile): SpatialPhysics {
  // Reliability → viscosity
  // RELIABLE packets stay in order → high viscosity (thick, ordered flow)
  // BEST_EFFORT packets can drop   → low viscosity  (thin, turbulent flow)
  const viscosity = qos.reliability === 'RELIABLE' ? 0.02 : 0.001;

  // Durability → boundary friction
  // TRANSIENT_LOCAL retains last messages → sticky walls (high friction)
  // VOLATILE forgets immediately          → slippery walls (low friction)
  const boundaryFriction = qos.durability === 'TRANSIENT_LOCAL' ? 0.95 : 0.30;

  // History depth → trail length (capped at 30)
  const trailLength = Math.min(qos.historyDepth * 3, 30);

  // MPS → emitter rate
  // 0 means unlimited → render as max burst (120 particles/s)
  const emitterRate = qos.mps === 0 ? 120.0 : Math.min(qos.mps * 1.5, 120.0);

  // Priority 0-3 → force multiplier 1.0-2.5 (linear interpolation)
  const forceMultiplier = 1.0 + qos.priority * 0.5;

  return { viscosity, boundaryFriction, trailLength, emitterRate, forceMultiplier };
}

// ---------------------------------------------------------------------------
// Apollo CyberRT QoS profiles — all 8 entries from upstream source
// Source: apollo/cyber/transport/qos/qos_profile_conf.cc
// ---------------------------------------------------------------------------
export const APOLLO_PROFILES: Record<string, QoSProfile> = {

  // DEFAULT: 通用控制消息通道，可靠交付，浅队列（depth 1），无速率限制
  DEFAULT: {
    reliability:  'RELIABLE',    // 保证消息不丢失，适合低频控制指令
    durability:   'VOLATILE',    // 不保留历史，晚加入节点不补发
    historyDepth: 1,             // 只保留最新 1 条消息
    mps:          0,             // 0 = 不限速
    priority:     1,             // 普通优先级
  },

  // SENSOR_DATA: 高频传感器流（激光雷达/摄像头/毫米波雷达），允许偶发丢包
  SENSOR_DATA: {
    reliability:  'BEST_EFFORT', // 尽力而为，丢帧可接受，降低延迟
    durability:   'VOLATILE',    // 旧帧无意义，不保留历史
    historyDepth: 5,             // 缓存最近 5 帧以平滑突发
    mps:          0,             // 不限速，由传感器驱动决定帧率
    priority:     0,             // 最低优先级，带宽紧张时可被抢占
  },

  // PARAMETERS: 参数服务器通道，晚加入节点需要获取完整参数历史
  PARAMETERS: {
    reliability:  'RELIABLE',         // 参数变更不允许丢失
    durability:   'TRANSIENT_LOCAL',  // 本地持久化，晚加入节点可补发全量历史
    historyDepth: 1000,               // 保留最近 1000 条参数记录
    mps:          0,                  // 不限速
    priority:     2,                  // 较高优先级，确保配置及时下发
  },

  // SERVICES_DEFAULT: RPC 服务调用通道（请求-响应模式）
  SERVICES_DEFAULT: {
    reliability:  'RELIABLE',    // 服务调用必须可靠，不允许丢失请求
    durability:   'VOLATILE',    // 会话外历史无意义，不保留
    historyDepth: 10,            // 缓冲 10 条并发请求
    mps:          0,             // 不限速
    priority:     2,             // 与参数通道同等优先级
  },

  // PARAM_EVENT: 参数变更事件总线，订阅者不应错过任何配置更新
  PARAM_EVENT: {
    reliability:  'RELIABLE',         // 事件不允许丢失
    durability:   'TRANSIENT_LOCAL',  // 持久化，晚加入节点可回溯变更历史
    historyDepth: 1000,               // 保留最近 1000 条事件
    mps:          0,                  // 不限速
    priority:     2,                  // 与参数通道同等优先级
  },

  // SYSTEM_DEFAULT: 系统级基础设施通道（对标 ROS 2 /rosout 风格）
  SYSTEM_DEFAULT: {
    reliability:  'RELIABLE',    // 系统内部消息可靠交付
    durability:   'VOLATILE',    // 不保留历史，仅关注实时状态
    historyDepth: 1,             // 极浅队列，只保留最新状态
    mps:          0,             // 不限速
    priority:     1,             // 普通优先级，与 DEFAULT 对齐
  },

  // TF_STATIC: 静态坐标系变换，晚加入节点必须能接收到最新变换
  TF_STATIC: {
    reliability:  'RELIABLE',         // 变换矩阵不允许丢失
    durability:   'TRANSIENT_LOCAL',  // 持久化，保证晚加入节点收到广播
    historyDepth: 1,                  // 静态变换只需最新一条即可
    mps:          0,                  // 不限速
    priority:     1,                  // 普通优先级
  },

  // TOPO_CHANGE: 图拓扑变更事件（节点/边的增删），最高优先级
  TOPO_CHANGE: {
    reliability:  'RELIABLE',         // 拓扑变更不允许丢失
    durability:   'TRANSIENT_LOCAL',  // 持久化，晚加入节点可重建当前拓扑
    historyDepth: 10,                 // 保留最近 10 次拓扑变更
    mps:          0,                  // 不限速
    priority:     3,                  // 最高优先级，确保路由信息最先送达
  },
};

// ---------------------------------------------------------------------------
// Human-readable descriptions for UI tooltips / legend panels
// ---------------------------------------------------------------------------
export const PROFILE_DESCRIPTIONS: Record<string, string> = {
  DEFAULT:
    'General-purpose channel. Reliable delivery, volatile history, ' +
    'shallow queue (depth 1). Suitable for infrequent control messages.',
  SENSOR_DATA:
    'High-frequency sensor streams (lidar/camera/radar). Best-effort ' +
    'delivery tolerates occasional drops; volatile history discards stale frames.',
  PARAMETERS:
    'Parameter server channel. Reliable + transient-local so late-joining ' +
    'nodes receive the full parameter history on connect.',
  SERVICES_DEFAULT:
    'RPC-style service calls. Reliable, volatile, shallow queue; ' +
    'each request expects a matched response within the session.',
  PARAM_EVENT:
    'Parameter-change event bus. Same reliability/durability as PARAMETERS ' +
    'so subscribers never miss a configuration update.',
  SYSTEM_DEFAULT:
    'System-level fallback profile. Matches Apollo internal infra channels; ' +
    'reliable, volatile, depth 1.',
  TF_STATIC:
    'Static coordinate-frame transforms. Transient-local depth-1 ensures ' +
    'any node that joins after broadcast still receives the transform.',
  TOPO_CHANGE:
    'Graph topology updates (node/edge add-remove). Highest priority (3), ' +
    'transient-local so late joiners reconstruct the current topology.',
};
