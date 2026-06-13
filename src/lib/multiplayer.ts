/**
 * multiplayer.ts — Real-time multiplayer networking layer
 *
 * Covers:
 *   SocketConnection   — low-level WebSocket wrapper (auto-reconnect, ping/pong, binary)
 *   GameCenterPlayer   — player identity & state snapshot
 *   GameCenterRoom     — room membership, lock/unlock, shared state CRDT-lite
 *   GameCenter         — server discovery, matchmaking, room registry
 *   Multiplayer        — high-level facade (join/leave/broadcast/rpc/presence)
 *
 * Wire protocol: newline-delimited JSON over WebSocket.
 * Binary blobs use ArrayBuffer frames (opcode prefix byte).
 */

// ─── Types ────────────────────────────────────────────────────────────────────

export type PlayerId  = string;
export type RoomId    = string;
export type MessageId = string;

export interface PlayerMeta {
  id:          PlayerId;
  displayName: string;
  avatarUrl?:  string;
  role:        'host' | 'guest';
  joinedAt:    number; // epoch ms
  latency:     number; // ms RTT
}

export interface RoomMeta {
  id:         RoomId;
  name:       string;
  capacity:   number;
  playerCount: number;
  locked:     boolean;
  mode:       string;
  createdAt:  number;
}

export interface NetworkMessage<T = unknown> {
  id:       MessageId;
  type:     string;
  senderId: PlayerId;
  roomId?:  RoomId;
  payload:  T;
  ts:       number;
}

export interface MultiplayerOptions {
  serverUrl:        string;
  reconnect?:       boolean;
  reconnectDelay?:  number; // ms
  maxReconnects?:   number;
  pingInterval?:    number; // ms
  binaryThreshold?: number; // bytes; payloads above this go binary
}

export interface RpcResult<T = unknown> {
  ok:    boolean;
  data?: T;
  error?: string;
}

// ─── SocketConnection ────────────────────────────────────────────────────────

type SocketEvent = 'open' | 'close' | 'error' | 'message' | 'binary' | 'reconnect';
type SocketHandler<T = unknown> = (data: T) => void;

export class SocketConnection extends EventTarget {
  private _url:           string;
  private _ws:            WebSocket | null = null;
  private _reconnect:     boolean;
  private _reconnectDelay:number;
  private _maxReconnects: number;
  private _pingInterval:  number;
  private _reconnectCount:number = 0;
  private _pingTimer:     ReturnType<typeof setInterval> | null = null;
  private _pendingPing:   number | null = null;
  private _latency:       number = 0;
  private _open:          boolean = false;
  private _sendQueue:     (string | ArrayBuffer)[] = [];

  constructor(url: string, opts: Partial<MultiplayerOptions> = {}) {
    super();
    this._url            = url;
    this._reconnect      = opts.reconnect      ?? true;
    this._reconnectDelay = opts.reconnectDelay ?? 2000;
    this._maxReconnects  = opts.maxReconnects  ?? 10;
    this._pingInterval   = opts.pingInterval   ?? 5000;
  }

  get latency():   number  { return this._latency; }
  get isOpen():    boolean { return this._open; }
  get url():       string  { return this._url; }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this._ws && this._open) { resolve(); return; }

      const ws = new WebSocket(this._url);
      ws.binaryType = 'arraybuffer';
      this._ws = ws;

      ws.onopen = () => {
        this._open = true;
        this._reconnectCount = 0;
        this._drainQueue();
        this._startPing();
        this.dispatchEvent(new Event('open'));
        resolve();
      };

      ws.onclose = (ev) => {
        this._open = false;
        this._stopPing();
        this.dispatchEvent(new CloseEvent('close', { code: ev.code, reason: ev.reason }));
        if (this._reconnect && this._reconnectCount < this._maxReconnects) {
          this._reconnectCount++;
          setTimeout(() => {
            this.dispatchEvent(new CustomEvent('reconnect', { detail: { attempt: this._reconnectCount } }));
            this.connect().catch(() => {});
          }, this._reconnectDelay * Math.min(this._reconnectCount, 4));
        }
      };

      ws.onerror = (ev) => {
        this.dispatchEvent(new ErrorEvent('error', { error: ev }));
        reject(ev);
      };

      ws.onmessage = (ev) => {
        if (ev.data instanceof ArrayBuffer) {
          this.dispatchEvent(new CustomEvent('binary', { detail: ev.data }));
          return;
        }
        const raw = ev.data as string;
        // Handle pong
        if (raw.startsWith('__pong__')) {
          if (this._pendingPing !== null) {
            this._latency = Date.now() - this._pendingPing;
            this._pendingPing = null;
          }
          return;
        }
        try {
          const msg: NetworkMessage = JSON.parse(raw);
          this.dispatchEvent(new CustomEvent('message', { detail: msg }));
        } catch {
          this.dispatchEvent(new CustomEvent('message', { detail: raw }));
        }
      };
    });
  }

  send(data: string | ArrayBuffer): void {
    if (this._open && this._ws) {
      this._ws.send(data);
    } else {
      this._sendQueue.push(data);
    }
  }

  sendJSON<T>(msg: T): void {
    this.send(JSON.stringify(msg));
  }

  close(code = 1000, reason = 'client closed'): void {
    this._reconnect = false;
    this._ws?.close(code, reason);
  }

  private _drainQueue(): void {
    while (this._sendQueue.length && this._open && this._ws) {
      this._ws.send(this._sendQueue.shift()!);
    }
  }

  private _startPing(): void {
    this._pingTimer = setInterval(() => {
      if (!this._open) return;
      this._pendingPing = Date.now();
      this._ws?.send('__ping__');
    }, this._pingInterval);
  }

  private _stopPing(): void {
    if (this._pingTimer !== null) { clearInterval(this._pingTimer); this._pingTimer = null; }
  }

  on<T = unknown>(event: SocketEvent, handler: SocketHandler<T>): this {
    this.addEventListener(event, (e) => {
      handler((e as CustomEvent<T>).detail ?? (e as unknown) as T);
    });
    return this;
  }

  dispose(): void {
    this._reconnect = false;
    this._stopPing();
    this._ws?.close(1000, 'disposed');
    this._sendQueue = [];
  }
}

// ─── GameCenterPlayer ────────────────────────────────────────────────────────

export class GameCenterPlayer extends EventTarget {
  private _meta:      PlayerMeta;
  private _state:     Record<string, unknown> = {};
  private _stateSeq:  number = 0;

  constructor(meta: PlayerMeta) {
    super();
    this._meta = { ...meta };
  }

  get id():          PlayerId    { return this._meta.id; }
  get displayName(): string      { return this._meta.displayName; }
  get role():        string      { return this._meta.role; }
  get latency():     number      { return this._meta.latency; }
  get meta():        Readonly<PlayerMeta> { return this._meta; }
  get state():       Readonly<Record<string, unknown>> { return this._state; }
  get stateSeq():    number      { return this._stateSeq; }

  updateMeta(patch: Partial<PlayerMeta>): void {
    Object.assign(this._meta, patch);
    this.dispatchEvent(new CustomEvent('metachange', { detail: this._meta }));
  }

  /** Apply a state patch (CRDT-lite: only apply if seq is newer). */
  applyState(patch: Record<string, unknown>, seq: number): boolean {
    if (seq <= this._stateSeq) return false;
    Object.assign(this._state, patch);
    this._stateSeq = seq;
    this.dispatchEvent(new CustomEvent('statechange', { detail: { patch, seq } }));
    return true;
  }

  toJSON(): PlayerMeta & { state: Record<string, unknown> } {
    return { ...this._meta, state: { ...this._state } };
  }
}

// ─── GameCenterRoom ──────────────────────────────────────────────────────────

export class GameCenterRoom extends EventTarget {
  private _meta:    RoomMeta;
  private _players: Map<PlayerId, GameCenterPlayer> = new Map();
  private _state:   Record<string, unknown> = {};
  private _stateSeq:number = 0;

  constructor(meta: RoomMeta) {
    super();
    this._meta = { ...meta };
  }

  get id():          RoomId    { return this._meta.id; }
  get name():        string    { return this._meta.name; }
  get capacity():    number    { return this._meta.capacity; }
  get locked():      boolean   { return this._meta.locked; }
  get mode():        string    { return this._meta.mode; }
  get isFull():      boolean   { return this._players.size >= this._meta.capacity; }
  get playerCount(): number    { return this._players.size; }
  get meta():        Readonly<RoomMeta> { return this._meta; }
  get players():     ReadonlyMap<PlayerId, GameCenterPlayer> { return this._players; }
  get state():       Readonly<Record<string, unknown>> { return this._state; }

  addPlayer(player: GameCenterPlayer): void {
    if (this._players.has(player.id)) return;
    this._players.set(player.id, player);
    this._meta.playerCount = this._players.size;
    this.dispatchEvent(new CustomEvent('playerjoin', { detail: player }));
  }

  removePlayer(id: PlayerId): GameCenterPlayer | undefined {
    const p = this._players.get(id);
    if (!p) return;
    this._players.delete(id);
    this._meta.playerCount = this._players.size;
    this.dispatchEvent(new CustomEvent('playerleave', { detail: p }));
    return p;
  }

  getPlayer(id: PlayerId): GameCenterPlayer | undefined {
    return this._players.get(id);
  }

  updateMeta(patch: Partial<RoomMeta>): void {
    Object.assign(this._meta, patch);
    this.dispatchEvent(new CustomEvent('metachanged', { detail: this._meta }));
  }

  applySharedState(patch: Record<string, unknown>, seq: number): boolean {
    if (seq <= this._stateSeq) return false;
    Object.assign(this._state, patch);
    this._stateSeq = seq;
    this.dispatchEvent(new CustomEvent('statechange', { detail: { patch, seq } }));
    return true;
  }

  lock():   void { this._meta.locked = true;  this.dispatchEvent(new Event('lock'));   }
  unlock(): void { this._meta.locked = false; this.dispatchEvent(new Event('unlock')); }

  toJSON(): RoomMeta & { players: ReturnType<GameCenterPlayer['toJSON']>[] } {
    return {
      ...this._meta,
      players: Array.from(this._players.values()).map(p => p.toJSON()),
    };
  }
}

// ─── GameCenter ───────────────────────────────────────────────────────────────

export interface GameCenterConfig {
  serverUrl:   string;
  gameId:      string;
  region?:     string;
  version?:    string;
}

export class GameCenter extends EventTarget {
  private _config:    GameCenterConfig;
  private _socket:    SocketConnection | null = null;
  private _rooms:     Map<RoomId, GameCenterRoom> = new Map();
  private _localId:   PlayerId | null = null;

  constructor(config: GameCenterConfig) {
    super();
    this._config = config;
  }

  get localPlayerId(): PlayerId | null { return this._localId; }
  get rooms(): ReadonlyMap<RoomId, GameCenterRoom> { return this._rooms; }

  async connect(displayName: string): Promise<PlayerId> {
    this._socket = new SocketConnection(this._config.serverUrl, { reconnect: true });

    this._socket.on<NetworkMessage>('message', (msg) => this._onMessage(msg));

    await this._socket.connect();

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('GameCenter auth timeout')), 8000);

      const handler = (e: Event) => {
        const msg = (e as CustomEvent<NetworkMessage>).detail;
        if (msg.type === 'auth_ok') {
          clearTimeout(timeout);
          this._localId = msg.payload as PlayerId;
          this._socket!.removeEventListener('message', handler);
          this.dispatchEvent(new CustomEvent('connected', { detail: this._localId }));
          resolve(this._localId);
        }
      };
      this._socket!.addEventListener('message', handler);

      this._socket!.sendJSON({
        type:    'auth',
        payload: { displayName, gameId: this._config.gameId, version: this._config.version },
      });
    });
  }

  async listRooms(mode?: string): Promise<RoomMeta[]> {
    return this._rpc<RoomMeta[]>('list_rooms', { mode });
  }

  async createRoom(name: string, capacity: number, mode: string): Promise<GameCenterRoom> {
    const meta = await this._rpc<RoomMeta>('create_room', { name, capacity, mode });
    const room = new GameCenterRoom(meta);
    this._rooms.set(room.id, room);
    return room;
  }

  async joinRoom(roomId: RoomId): Promise<GameCenterRoom> {
    const meta = await this._rpc<RoomMeta & { players: PlayerMeta[] }>('join_room', { roomId });
    let room = this._rooms.get(roomId);
    if (!room) { room = new GameCenterRoom(meta); this._rooms.set(roomId, room); }
    for (const pm of meta.players) {
      if (!room.getPlayer(pm.id)) room.addPlayer(new GameCenterPlayer(pm));
    }
    this.dispatchEvent(new CustomEvent('roomjoined', { detail: room }));
    return room;
  }

  async leaveRoom(roomId: RoomId): Promise<void> {
    await this._rpc('leave_room', { roomId });
    const room = this._rooms.get(roomId);
    if (room) {
      this._rooms.delete(roomId);
      this.dispatchEvent(new CustomEvent('roomleft', { detail: room }));
    }
  }

  /** Fire-and-forget broadcast to all players in room. */
  broadcast<T>(roomId: RoomId, type: string, payload: T): void {
    this._socket?.sendJSON({ type: 'broadcast', roomId, subtype: type, payload, ts: Date.now() });
  }

  /** Reliable RPC to server (awaits response). */
  private _rpc<T>(method: string, params: unknown): Promise<T> {
    return new Promise((resolve, reject) => {
      const id = `rpc_${Date.now()}_${Math.random().toString(36).slice(2)}`;
      const timeout = setTimeout(() => reject(new Error(`RPC timeout: ${method}`)), 10000);

      const handler = (e: Event) => {
        const msg = (e as CustomEvent<NetworkMessage<RpcResult<T>>>).detail;
        if (msg.id === id && msg.type === 'rpc_result') {
          clearTimeout(timeout);
          this._socket!.removeEventListener('message', handler);
          if (msg.payload.ok) resolve(msg.payload.data as T);
          else reject(new Error(msg.payload.error ?? 'RPC error'));
        }
      };
      this._socket?.addEventListener('message', handler);
      this._socket?.sendJSON({ id, type: 'rpc', method, params, ts: Date.now() });
    });
  }

  private _onMessage(msg: NetworkMessage): void {
    switch (msg.type) {
      case 'player_joined': {
        const room = this._rooms.get(msg.roomId!);
        if (room) room.addPlayer(new GameCenterPlayer(msg.payload as PlayerMeta));
        break;
      }
      case 'player_left': {
        const room = this._rooms.get(msg.roomId!);
        if (room) room.removePlayer(msg.payload as PlayerId);
        break;
      }
      case 'broadcast': {
        const room = this._rooms.get(msg.roomId!);
        if (room) room.dispatchEvent(new CustomEvent('message', { detail: msg }));
        this.dispatchEvent(new CustomEvent('message', { detail: msg }));
        break;
      }
      case 'room_state': {
        const room = this._rooms.get(msg.roomId!);
        const d = msg.payload as { patch: Record<string, unknown>; seq: number };
        room?.applySharedState(d.patch, d.seq);
        break;
      }
      case 'player_state': {
        const room = this._rooms.get(msg.roomId!);
        const d = msg.payload as { playerId: PlayerId; patch: Record<string, unknown>; seq: number };
        room?.getPlayer(d.playerId)?.applyState(d.patch, d.seq);
        break;
      }
    }
    this.dispatchEvent(new CustomEvent('rawmessage', { detail: msg }));
  }

  disconnect(): void {
    this._socket?.dispose();
    this._socket = null;
    this._rooms.clear();
    this._localId = null;
  }
}

// ─── Multiplayer ──────────────────────────────────────────────────────────────

export interface MultiplayerConfig extends MultiplayerOptions {
  localPlayerId:   PlayerId;
  localPlayerName: string;
  roomId?:         RoomId;
}

/**
 * High-level facade that wraps SocketConnection for
 * peer-to-peer-style messaging without a full GameCenter server.
 */
export class Multiplayer extends EventTarget {
  private _socket:     SocketConnection;
  private _config:     MultiplayerConfig;
  private _players:    Map<PlayerId, GameCenterPlayer> = new Map();
  private _localPlayer:GameCenterPlayer;
  private _msgSeq:     number = 0;

  constructor(config: MultiplayerConfig) {
    super();
    this._config = config;
    this._socket = new SocketConnection(config.serverUrl, config);
    this._localPlayer = new GameCenterPlayer({
      id:          config.localPlayerId,
      displayName: config.localPlayerName,
      role:        'host',
      joinedAt:    Date.now(),
      latency:     0,
    });
    this._players.set(config.localPlayerId, this._localPlayer);
    this._socket.on<NetworkMessage>('message', msg => this._dispatch(msg));
    this._socket.on<void>('open', () => this._onOpen());
  }

  get localPlayer(): GameCenterPlayer { return this._localPlayer; }
  get players(): ReadonlyMap<PlayerId, GameCenterPlayer> { return this._players; }
  get socket(): SocketConnection { return this._socket; }

  async connect(): Promise<void> {
    await this._socket.connect();
  }

  private _onOpen(): void {
    // Announce presence
    this._send('presence', {
      id:          this._config.localPlayerId,
      displayName: this._config.localPlayerName,
      roomId:      this._config.roomId,
    });
    this.dispatchEvent(new Event('connected'));
  }

  /** Broadcast a message to all peers in the room. */
  broadcast<T>(type: string, payload: T): void {
    this._send(type, payload);
  }

  /** Send to a specific player (server routes it). */
  sendTo<T>(targetId: PlayerId, type: string, payload: T): void {
    this._socket.sendJSON<NetworkMessage>({
      id:       this._nextId(),
      type,
      senderId: this._config.localPlayerId,
      roomId:   this._config.roomId,
      payload:  { ...( payload as object ), _targetId: targetId },
      ts:       Date.now(),
    });
  }

  /** Update local player state and broadcast diff. */
  setLocalState(patch: Record<string, unknown>): void {
    this._msgSeq++;
    this._localPlayer.applyState(patch, this._msgSeq);
    this._send('player_state', {
      playerId: this._config.localPlayerId,
      patch,
      seq: this._msgSeq,
    });
  }

  /** Send binary data (e.g. compressed snapshot). */
  sendBinary(buffer: ArrayBuffer): void {
    this._socket.send(buffer);
  }

  private _send<T>(type: string, payload: T): void {
    this._socket.sendJSON<NetworkMessage>({
      id:       this._nextId(),
      type,
      senderId: this._config.localPlayerId,
      roomId:   this._config.roomId,
      payload,
      ts:       Date.now(),
    });
  }

  private _nextId(): MessageId {
    return `${this._config.localPlayerId}_${Date.now()}_${this._msgSeq++}`;
  }

  private _dispatch(msg: NetworkMessage): void {
    switch (msg.type) {
      case 'presence': {
        const d = msg.payload as PlayerMeta;
        if (!this._players.has(d.id)) {
          const p = new GameCenterPlayer(d);
          this._players.set(d.id, p);
          this.dispatchEvent(new CustomEvent('playerjoin', { detail: p }));
        }
        break;
      }
      case 'player_state': {
        const d = msg.payload as { playerId: PlayerId; patch: Record<string, unknown>; seq: number };
        this._players.get(d.playerId)?.applyState(d.patch, d.seq);
        break;
      }
      case 'disconnect': {
        const id = msg.payload as PlayerId;
        const p = this._players.get(id);
        if (p) {
          this._players.delete(id);
          this.dispatchEvent(new CustomEvent('playerleave', { detail: p }));
        }
        break;
      }
    }
    this.dispatchEvent(new CustomEvent('message', { detail: msg }));
  }

  disconnect(): void {
    this._send('disconnect', this._config.localPlayerId);
    this._socket.dispose();
  }
}
