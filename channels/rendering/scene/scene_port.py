import os, sys, json, math
from typing import Any, Optional
from channels.rendering.species.species_port import _species_to_index
from channels.rendering.decoration.decoration_extra import _SPECIES_INDEX_TO_COLOUR

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)



# =============================================================================
# [SceneCaptureRendering] AstroCellCaptureMode + AstroCellCaptureProcessor
# =============================================================================

class AstroCellCaptureMode:
    """
    Python equivalent of FSceneCapturePS::ESourceMode enum (8 values).

    Maps UE5 ESceneCaptureSource constants to Astro channel modes that
    determine which data is sampled from the rendered cell scene.

    鲁迅式：源模式是摄像头的目的——
    你想捕捉颜色、深度、还是法线？
    目的不同，捕捉的手段和代价也不同。
    """
    COLOR_AND_OPACITY   = 0   # SCS_SceneColorHDR
    COLOR_NO_ALPHA      = 1   # SCS_SceneColorHDRNoAlpha
    COLOR_AND_DEPTH     = 2   # SCS_SceneColorSceneDepth
    SCENE_DEPTH         = 3   # SCS_SceneDepth
    DEVICE_DEPTH        = 4   # SCS_DeviceDepth
    NORMAL              = 5   # SCS_Normal
    BASE_COLOR          = 6   # SCS_BaseColor
    COLOR_ONE_ALPHA     = 7   # SCS_SceneColorHDRNoAlpha for reflection capture

    _SOURCE_TO_MODE = {
        "color_opacity":    0,
        "color_no_alpha":   1,
        "color_depth":      2,
        "depth":            3,
        "device_depth":     4,
        "normal":           5,
        "base_color":       6,
        "color_one_alpha":  7,
    }

    @classmethod
    def from_source_name(cls, name: str) -> int:
        return cls._SOURCE_TO_MODE.get(name, 0)





def _should_compile_capture_permutation(
    source_mode: int,
    use_128bit_rt: bool,
    requires_explicit_128bit: bool,
) -> bool:
    """
    Mirrors FSceneCapturePS::ShouldCompilePermutation():
        return (!PermutationVector.Get<FEnable128BitRT>()
                || bPlatformRequiresExplicit128bitRT);

    In the Astro context: 128-bit RT is approximated by float16 SVG colour
    channels (not needed for standard uint8 output).

    鲁迅式：应该编译的 permutation 才编译——
    无用的组合是浪费，删除它们是一种诚实。
    """
    if use_128bit_rt and not requires_explicit_128bit:
        return False
    return True





def _get_capture_permutation(
    source_name: str,
    use_128bit_rt:          bool = False,
    forward_shading:        bool = False,
    is_reflection_capture:  bool = False,
) -> dict:
    """
    Compute capture permutation parameters.

    Mirrors FSceneCapturePS::GetPermutationVector():
        Maps ESceneCaptureSource → ESourceMode.
        Handles forward-shading override for Normal/BaseColor modes.
        Returns {source_mode, use_128bit_rt}.

    鲁迅式：Permutation 是现实的分叉——每一个旗标都是一条路，
    组合爆炸是工程师的噩梦，也是用户功能的保障。
    """
    mode = AstroCellCaptureMode.from_source_name(source_name)

    # Reflection capture: NoAlpha → ColorOneAlpha
    if is_reflection_capture and mode == AstroCellCaptureMode.COLOR_NO_ALPHA:
        mode = AstroCellCaptureMode.COLOR_ONE_ALPHA

    # Forward shading override: Normal/BaseColor → ColorAndOpacity
    if forward_shading and mode in (
        AstroCellCaptureMode.NORMAL, AstroCellCaptureMode.BASE_COLOR
    ):
        mode = AstroCellCaptureMode.COLOR_AND_OPACITY

    return {"source_mode": mode, "use_128bit_rt": use_128bit_rt}





class AstroCellCaptureProcessor:
    """
    Python equivalent of the SceneCaptureRendering pipeline.

    Captures the current cell scene state into a «render target» dict,
    supporting the 8 ESourceMode channel configurations from FSceneCapturePS.

    Two primary entry points mirror the UE5 C++ functions:
      capture_scene()        → CaptureSceneToRenderTarget() analog
      copy_capture_to_target()→ CopyCaptureToTarget() / UpdateSceneCaptureContents()

    鲁迅式：场景捕获是镜子——它把当前世界的状态定格为一张快照，
    供反射、后处理、UI 叠加等系统消费。
    镜子不创造，但它记录；记录本身，便是一种价值。
    """

    def __init__(self,
                 source_name:          str   = "color_opacity",
                 use_128bit_rt:         bool  = False,
                 forward_shading:       bool  = False,
                 allow_main_renderer:   bool  = ASTRO_CAPTURE_ALLOW_MAIN_RENDERER,
                 cube_single_pass:      bool  = ASTRO_CAPTURE_CUBE_SINGLE_PASS) -> None:
        self._permutation = _get_capture_permutation(
            source_name, use_128bit_rt, forward_shading,
        )
        self.allow_main_renderer = allow_main_renderer
        self.cube_single_pass    = cube_single_pass
        self._render_target: dict = {}

    @property
    def source_mode(self) -> int:
        return self._permutation["source_mode"]

    def capture_scene(
        self,
        cell_entries:  list,
        depth_manifest: dict,
        viewport_w:    float = 1200.0,
        viewport_h:    float = 900.0,
    ) -> dict:
        """
        Capture the scene into a render-target dict.

        Mirrors CaptureSceneToRenderTarget() / UpdateSceneCaptureContents():
          - If allow_main_renderer and source_mode supports it:
              render as part of the main renderer (inline capture path).
          - Otherwise:
              render as independent scene (separate capture path).

        For each capture mode, different data channels are populated:
          COLOR_AND_OPACITY → rgba per cell (fill + opacity)
          SCENE_DEPTH       → normalised Z per cell
          NORMAL            → surface normal vector per cell
          BASE_COLOR        → species primary colour (no lighting)
          COLOR_ONE_ALPHA   → reflection capture alpha=1 convention

        Returns the populated render target dict.

        鲁迅式：捕获场景需要代价——你每多渲染一次，GPU 就多工作一次。
        AllowRenderInMainRenderer 是一种妥协：如果主渲染器能顺路帮你做，
        何必另起炉灶？
        """
        mode = self.source_mode
        render_target: dict = {
            "source_mode":   mode,
            "viewport":      {"w": viewport_w, "h": viewport_h},
            "cells":         {},
        }

        for entry in cell_entries:
            cid     = entry.get("cell_id", "")
            species = entry.get("species", "")
            bbox    = entry.get("bbox", {})
            opacity = float(entry.get("opacity", 1.0))
            z       = float(bbox.get("z", 0))

            sp_idx = _species_to_index(species)
            colour = _SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0])

            if mode == AstroCellCaptureMode.COLOR_AND_OPACITY:
                cell_data = {
                    "r": colour[0] / 255.0, "g": colour[1] / 255.0,
                    "b": colour[2] / 255.0, "a": opacity,
                }
            elif mode == AstroCellCaptureMode.COLOR_NO_ALPHA:
                cell_data = {
                    "r": colour[0] / 255.0, "g": colour[1] / 255.0,
                    "b": colour[2] / 255.0, "a": 1.0,
                }
            elif mode == AstroCellCaptureMode.SCENE_DEPTH:
                depth = depth_manifest.get("depth_channel", {}).get(cid, 1.0)
                cell_data = {"depth": depth}
            elif mode == AstroCellCaptureMode.DEVICE_DEPTH:
                # Device depth = 1 - scene_depth (UE5 reversed-Z convention)
                depth = depth_manifest.get("depth_channel", {}).get(cid, 1.0)
                cell_data = {"device_depth": 1.0 - depth}
            elif mode == AstroCellCaptureMode.NORMAL:
                # Surface normals: cells face toward viewer (+Z)
                cell_data = {"nx": 0.0, "ny": 0.0, "nz": 1.0}
            elif mode == AstroCellCaptureMode.BASE_COLOR:
                cell_data = {
                    "r": colour[0] / 255.0,
                    "g": colour[1] / 255.0,
                    "b": colour[2] / 255.0,
                }
            elif mode == AstroCellCaptureMode.COLOR_ONE_ALPHA:
                # Reflection capture: alpha forced to 1 (mirrors reflection path)
                cell_data = {
                    "r": colour[0] / 255.0, "g": colour[1] / 255.0,
                    "b": colour[2] / 255.0, "a": 1.0,
                }
            else:
                cell_data = {"r": 0.0, "g": 0.0, "b": 0.0, "a": opacity}

            cell_data["z"]       = z
            cell_data["species"] = species
            render_target["cells"][cid] = cell_data

        self._render_target = render_target
        return render_target

    def copy_capture_to_target(
        self,
        target_path: str,
    ) -> None:
        """
        Persist the captured render target to the channel filesystem.

        Mirrors CopyCaptureToTarget() / the RDG pass that blits the
        scene capture result into the final render target texture.

        鲁迅式：数据不写出便等于不存在——
        捕获的每一帧都需要落地为文件，才能被后续系统消费。
        """
        if not self._render_target:
            return
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w") as _f:
            json.dump(self._render_target, _f, indent=2)
        print(
            f"[AstroCellCaptureProcessor] copy_capture_to_target: "
            f"mode={self.source_mode} cells={len(self._render_target.get('cells', {}))} "
            f"→ {target_path}",
            file=sys.stderr,
        )





def update_scene_capture_contents(
    cell_entries:    list,
    depth_manifest:  dict,
    capture_dir:     str,
    source_name:     str  = "color_opacity",
    is_reflection:   bool = False,
    viewport_w:      float = 1200.0,
    viewport_h:      float = 900.0,
) -> dict:
    """
    Top-level scene capture update.

    Mirrors UpdateSceneCaptureContents() — the primary entry point called
    each frame to refresh a SceneCaptureComponent2D's render target.

    Constructs an AstroCellCaptureProcessor, runs capture_scene(), and
    persists the result to capture_dir/scene_capture.json.

    @param is_reflection  When True switches to COLOR_ONE_ALPHA permutation
                          (reflection capture convention).
    @return               Capture render target dict.

    鲁迅式：UpdateSceneCaptureContents 是场景捕获的总调度——
    每帧一次，不多不少。频率是性能与精度之间的谈判结果。
    """
    effective_source = "color_one_alpha" if is_reflection else source_name
    processor = AstroCellCaptureProcessor(
        source_name=effective_source,
        is_reflection_capture=is_reflection if False else False,  # resolved above
    )
    rt = processor.capture_scene(cell_entries, depth_manifest, viewport_w, viewport_h)

    target_path = os.path.join(capture_dir, "scene_capture.json")
    processor.copy_capture_to_target(target_path)
    return rt


# =============================================================================
# [ReflectionEnvironmentCapture] AstroCellReflectionCaptureState + pipeline
# =============================================================================




@dataclass
class AstroCellReflectionCaptureState:
    """
    Python equivalent of the per-capture runtime state in
    ReflectionEnvironmentCapture.cpp.

    Tracks the timeslicing state (which «faces» have been rendered),
    the fade-in progress, and the accumulated capture data for one
    reflection capture probe.

    Reflection «faces» in 2-D → six Z-layer offsets sampled around the
    capture origin: +Z, -Z, +X, -X, +Y, -Y (cardinal directions mapped to
    Z-layer and XY offset combinations).

    鲁迅式：时分渲染是对时间的借贷——每帧还一点债，六帧还清，
    然后重新开始。债不能不还，只是分期而已。
    """
    capture_id:       str   = ""
    world_pos:        tuple = (0.0, 0.0, 0.0)
    influence_radius: float = 1000.0
    # Timeslice state: which face index [0..5] is rendered next
    current_face:     int   = 0
    # Number of faces rendered so far in this cycle
    faces_rendered:   int   = 0
    # Captured colour data per face: face_index → (r, g, b)
    face_data:        dict  = field(default_factory=dict)
    # Fade-in progress [0.0, 1.0] — mirrors CVarReflectionCaptureRuntimeFadeInTime
    fade_progress:    float = 0.0
    # Whether the full cube has been captured at least once this session
    is_complete:      bool  = False
    # Frame index of last update
    last_update_frame: int  = -1

    _FACE_OFFSETS = [
        ( 0,  0,  1),   # face 0: +Z (upward)
        ( 0,  0, -1),   # face 1: -Z (downward)
        ( 1,  0,  0),   # face 2: +X (right)
        (-1,  0,  0),   # face 3: -X (left)
        ( 0,  1,  0),   # face 4: +Y (forward)
        ( 0, -1,  0),   # face 5: -Y (backward)
    ]

    def faces_per_timeslice(self) -> int:
        """
        Number of faces to render per frame.
        Mirrors CVarReflectionCaptureRuntimeTimeslice (clamped to [1, 6]).
        If ASTRO_CAPTURE_CUBE_SINGLE_PASS: render all 6 in one frame.
        """
        if ASTRO_CAPTURE_CUBE_SINGLE_PASS:
            return 6
        return max(1, min(6, _REFL_TIMESLICE_FACES))

    def sample_face(
        self,
        face_index: int,
        cell_entries: list,
        depth_manifest: dict,
    ) -> tuple:
        """
        Sample average scene colour for one probe «face».

        Mirrors CaptureSceneToScratchCubemap() for a single face:
          1. Select cells within influence_radius × face direction half-space.
          2. Average their species colours weighted by proximity.
          3. Apply supersample factor (multiple passes → average).

        Returns (r, g, b) average colour for this face.

        鲁迅式：每个方向采样一次，六个方向合而为一——
        这是环境光的民主原则：四面八方的光照都有发言权。
        """
        if face_index >= len(self._FACE_OFFSETS):
            return (0.5, 0.5, 0.5)

        fx, fy, fz = self._FACE_OFFSETS[face_index]
        ox, oy, oz = self.world_pos

        # Collect cells in this face's half-space (dot(cell_dir, face_dir) > 0)
        r_sum = g_sum = b_sum = weight_sum = 0.0
        supersample = _clamp_supersample(_REFL_SUPERSAMPLE_FACTOR)

        for entry in cell_entries:
            bbox    = entry.get("bbox", {})
            cx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) * 0.5
            cy = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) * 0.5
            cz = float(bbox.get("z", 0)) * 100.0   # z-layer → world units

            # Direction from probe to cell
            dx, dy, dz = cx - ox, cy - oy, cz - oz
            dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6

            # Half-space test: dot(cell_dir, face_dir) > 0
            dot = (dx/dist)*fx + (dy/dist)*fy + (dz/dist)*fz
            if dot <= 0.0:
                continue

            # Distance weight: exponential falloff within influence_radius
            # Mirrors the per-probe weight in the C++ cubemap blend pass.
            if dist > self.influence_radius:
                continue

            weight = (1.0 - dist / self.influence_radius) * dot

            species = entry.get("species", "")
            sp_idx  = _species_to_index(species)
            colour  = _SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0])

            r_sum += colour[0] / 255.0 * weight * supersample
            g_sum += colour[1] / 255.0 * weight * supersample
            b_sum += colour[2] / 255.0 * weight * supersample
            weight_sum += weight * supersample

        if weight_sum > 1e-6:
            return (r_sum / weight_sum, g_sum / weight_sum, b_sum / weight_sum)
        # No cells in this face direction → ambient grey
        return (0.5, 0.5, 0.5)

    def tick(
        self,
        frame_index:   int,
        cell_entries:  list,
        depth_manifest: dict,
        fade_in_time:  float = 0.5,
    ) -> bool:
        """
        Advance the timeslice capture by one frame.

        Renders faces_per_timeslice() faces per call, cycling through the
        six face indices.  Once all 6 faces are complete, is_complete=True
        and fade_progress advances toward 1.0.

        Returns True if the full cubemap was completed this frame (or already
        complete and mode==Once).

        Mirrors the timeslice logic in UpdateReflectionCaptures():
            Render N faces per frame (CVarReflectionCaptureRuntimeTimeslice).
            When all 6 done: mark IsComplete, start fade-in.
            Mode=Once (1): stop after first complete cycle.
            Mode=Continuous (0): repeat indefinitely.

        鲁迅式：六个面，每帧还一面的债——
        不急，不乱，债总是会还清的，然后重新开始借贷。
        这就是时分渲染的生存哲学。
        """
        if self.is_complete and _REFL_RUNTIME_MODE == 1:
            # Once mode: already done — just advance fade
            self.fade_progress = min(1.0, self.fade_progress + (1.0 / max(fade_in_time * 60, 1)))
            return True

        faces_this_tick = self.faces_per_timeslice()
        for _ in range(faces_this_tick):
            face_idx = self.current_face % 6
            colour   = self.sample_face(face_idx, cell_entries, depth_manifest)
            self.face_data[face_idx] = colour
            self.current_face = (self.current_face + 1) % 6
            self.faces_rendered += 1

        self.last_update_frame = frame_index

        # Check if a full cycle is complete
        if len(self.face_data) == 6:
            self.is_complete = True
            self.fade_progress = min(1.0,
                self.fade_progress + (1.0 / max(fade_in_time * 60, 1))
            )
            return True

        return False

    def average_radiance(self) -> tuple:
        """
        Compute the average radiance across all captured faces.

        Mirrors the cubemap averaging used to derive the dominant probe
        colour for the StyleProbe blend step.  Returns (r, g, b) float tuple.

        鲁迅式：六个方向的平均，是公平，也是妥协——
        没有哪个方向比另一个更重要，所以平等权重，一人一票。
        """
        if not self.face_data:
            return (0.5, 0.5, 0.5)
        n  = len(self.face_data)
        r  = sum(v[0] for v in self.face_data.values()) / n
        g  = sum(v[1] for v in self.face_data.values()) / n
        b  = sum(v[2] for v in self.face_data.values()) / n
        return (r, g, b)

    def to_dict(self) -> dict:
        return {
            "capture_id":      self.capture_id,
            "world_pos":       self.world_pos,
            "influence_radius": self.influence_radius,
            "faces_rendered":  self.faces_rendered,
            "is_complete":     self.is_complete,
            "fade_progress":   round(self.fade_progress, 4),
            "last_update_frame": self.last_update_frame,
            "face_data":       {str(k): list(v) for k, v in self.face_data.items()},
            "average_radiance": list(self.average_radiance()),
        }





class AstroCellReflectionCaptureManager:
    """
    Python equivalent of the UpdateReflectionCaptures() pipeline.

    Maintains a registry of AstroCellReflectionCaptureState probes, sorted
    by distance to the viewer camera, and dispatches per-frame timeslice
    updates subject to the _REFL_BUDGET probe-count cap.

    Mirrors the C++ flow in UpdateReflectionCaptures():
      1. Sort active captures by distance (nearest first).
      2. Apply budget cap (skip distant probes if over limit).
      3. For each active probe: call capture_state.tick().
      4. Persist results to physics/reflection_captures.json channel.

    鲁迅式：反射捕获管理器是公平的排队系统——
    距离越近的探针优先更新，预算有限时远处的探针被搁置。
    这不是歧视，是资源分配的现实。
    """

    def __init__(self) -> None:
        self._captures: dict = {}   # capture_id → AstroCellReflectionCaptureState
        self._frame_index: int = 0
        self._channel_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "physics", "reflection_captures.json",
        )

    def register_capture(
        self,
        capture_id:       str,
        world_pos:        tuple,
        influence_radius: float = 1000.0,
    ) -> AstroCellReflectionCaptureState:
        """
        Register a new reflection capture probe.
        Mirrors AddReflectionCapture() / the dynamic runtime probe lifecycle.
        """
        state = AstroCellReflectionCaptureState(
            capture_id=capture_id,
            world_pos=world_pos,
            influence_radius=influence_radius,
        )
        self._captures[capture_id] = state
        return state

    def remove_capture(self, capture_id: str) -> None:
        """Deregister a capture probe — mirrors RemoveReflectionCapture()."""
        self._captures.pop(capture_id, None)

    def tick(
        self,
        viewer_pos:    tuple,
        cell_entries:  list,
        depth_manifest: dict,
        fade_in_time:  float = 0.5,
    ) -> dict:
        """
        Per-frame reflection capture update pass.

        Mirrors UpdateReflectionCaptures() dispatch:
          1. Sort captures by distance to viewer.
          2. Apply _REFL_BUDGET cap.
          3. Tick each active capture.
          4. Persist results.

        Returns per-frame stats dict.

        鲁迅式：每帧更新反射——不停地照镜子，
        不是虚荣，是为了让世界在镜中保持真实。
        """
        self._frame_index += 1
        vx, vy, vz = viewer_pos

        # Sort captures by distance to viewer (nearest first)
        def _dist(state: AstroCellReflectionCaptureState) -> float:
            ox, oy, oz = state.world_pos
            return math.sqrt((ox-vx)**2 + (oy-vy)**2 + (oz-vz)**2)

        sorted_captures = sorted(self._captures.values(), key=_dist)

        # Apply budget cap (0 = unlimited)
        budget = _REFL_BUDGET
        active = sorted_captures if budget == 0 else sorted_captures[:budget]

        completed_this_frame = 0
        for state in active:
            done = state.tick(self._frame_index, cell_entries, depth_manifest, fade_in_time)
            if done:
                completed_this_frame += 1

        # Persist to channel
        all_data = {cid: s.to_dict() for cid, s in self._captures.items()}
        try:
            os.makedirs(os.path.dirname(self._channel_path), exist_ok=True)
            with open(self._channel_path, "w") as _f:
                json.dump(all_data, _f, indent=2)
        except OSError as _e:
            print(
                f"[AstroCellReflectionCaptureManager] WARNING: "
                f"failed to persist captures: {_e}",
                file=sys.stderr,
            )

        stats = {
            "frame_index":          self._frame_index,
            "total_captures":       len(self._captures),
            "active_captures":      len(active),
            "completed_this_frame": completed_this_frame,
            "budget_cap":           budget,
            "supersample_factor":   _clamp_supersample(_REFL_SUPERSAMPLE_FACTOR),
            "timeslice_faces":      _REFL_TIMESLICE_FACES,
            "runtime_mode":         "once" if _REFL_RUNTIME_MODE == 1 else "continuous",
        }

        print(
            f"[AstroCellReflectionCaptureMgr] tick: "
            f"frame={self._frame_index} "
            f"captures={len(self._captures)} active={len(active)} "
            f"completed_this_frame={completed_this_frame}",
            file=sys.stderr,
        )
        return stats

    def get_probe_radiance(self, capture_id: str) -> tuple:
        """
        Return the average face radiance for a named probe.
        Used by the StyleProbe blending system as an alternative neighbourhood
        colour source when explicit neighbour cells are absent.
        """
        state = self._captures.get(capture_id)
        if state and state.is_complete:
            return state.average_radiance()
        return (0.5, 0.5, 0.5)


#: Module-level reflection capture manager singleton.
_ASTRO_REFLECTION_CAPTURE_MANAGER: AstroCellReflectionCaptureManager | None = None





def get_reflection_capture_manager() -> AstroCellReflectionCaptureManager:
    """
    Return the process-level reflection capture manager singleton.

    Mirrors the FScene::ReflectionSceneData lifetime — one manager per scene.

    鲁迅式：反射系统的单例是场景中唯一的真相来源——
    所有探针都向它汇报，所有消费者都向它查询。
    中央化不总是好事，但在反射系统中，一致性比自由更重要。
    """
    global _ASTRO_REFLECTION_CAPTURE_MANAGER
    if _ASTRO_REFLECTION_CAPTURE_MANAGER is None:
        _ASTRO_REFLECTION_CAPTURE_MANAGER = AstroCellReflectionCaptureManager()
    return _ASTRO_REFLECTION_CAPTURE_MANAGER





def capture_scene_to_scratch_cubemap(
    cell_id:     str,
    bbox:        dict,
    species:     str,
    all_bboxes:  dict,
    face_index:  int  = -1,   # -1 = all 6 faces; 0..5 = specific face (timeslice)
    supersample: int  = ASTRO_CAPTURE_SUPERSAMPLE,
) -> AstroCellCaptureState:
    """
    Capture scene radiance into the cell's cubemap scratch buffer.

    Mirrors CaptureSceneToScratchCubemap() which renders 6 cube faces into
    a temporary render target, including sky atmosphere, sky light, and
    foliage (if enabled).

    2-D adaptation:
        Each face is assigned a canonical direction in the 2-D SVG plane:
            Face 0 (+X): right-side neighbours
            Face 1 (-X): left-side neighbours
            Face 2 (+Y): bottom neighbours
            Face 3 (-Y): top neighbours
            Face 4 (+Z): higher-z-layer neighbours  (「上」)
            Face 5 (-Z): lower-z-layer neighbours   (「下」)
        For each face, we query the BVH (or all_bboxes) for neighbours in
        that half-plane, average their emissive colours weighted by solid
        angle, and write the result into the face data.

    The supersample factor is clamped to [MIN, MAX] as in C++ and applied
    as a weight boost for the Gaussian downsample kernel.

    Returns the updated AstroCellCaptureState for *cell_id*.

    鲁迅式：捕获是对现实的凝视——六个方向，不遗漏任何角落。
    但凝视需要勇气：某些方向可能什么都没有，而这本身也是信息。
    """
    supersample = max(ASTRO_CAPTURE_SUPERSAMPLE_MIN,
                      min(ASTRO_CAPTURE_SUPERSAMPLE_MAX, supersample))

    scene  = get_reflection_scene_data()
    slot   = find_or_allocate_cubemap_index(cell_id)
    if slot < 0:
        return AstroCellCaptureState(cell_id=cell_id, cubemap_index=-1)

    capture = scene.allocated_captures[cell_id]

    cx   = bbox.get("x", 0) + bbox.get("w", 80) / 2.0
    cy   = bbox.get("y", 0) + bbox.get("h", 50) / 2.0
    cz   = float(bbox.get("z", 3))
    hw   = bbox.get("w", 80) / 2.0
    hh   = bbox.get("h", 50) / 2.0

    # Face query planes: half-planes in (X, Y, Z) centred on cell
    face_filters = [
        lambda b, _cx=cx: b.get("x", 0) > _cx,           # +X: right
        lambda b, _cx=cx: b.get("x", 0) + b.get("w", 80) < _cx,  # -X: left
        lambda b, _cy=cy: b.get("y", 0) > _cy,           # +Y: below (SVG y grows down)
        lambda b, _cy=cy: b.get("y", 0) + b.get("h", 50) < _cy,  # -Y: above
        lambda b, _cz=cz: float(b.get("z", 3)) > _cz,   # +Z: higher layer
        lambda b, _cz=cz: float(b.get("z", 3)) < _cz,   # -Z: lower layer
    ]

    faces_to_capture = range(_CAPTURE_NUM_FACES) if face_index < 0 else [face_index]

    _EMISSIVE_CAPTURE_TABLE = {
        "cil-eye":         (0.55, 0.60, 0.90),
        "cil-bolt":        (0.95, 0.55, 0.10),
        "cil-vector":      (0.30, 0.70, 0.35),
        "cil-plus":        (0.25, 0.55, 0.90),
        "cil-arrow-right": (0.50, 0.60, 0.65),
        "cil-filter":      (0.60, 0.25, 0.75),
        "cil-code":        (0.30, 0.70, 0.35),
        "cil-layers":      (0.20, 0.55, 0.85),
        "cil-loop":        (0.90, 0.60, 0.15),
        "cil-graph":       (0.40, 0.50, 0.55),
    }

    for fi in faces_to_capture:
        face_filter_fn = face_filters[fi]
        # Collect neighbours in this face's half-plane
        contributors = []
        for other_id, obbox in all_bboxes.items():
            if other_id == cell_id:
                continue
            if face_filter_fn(obbox):
                sp = obbox.get("species", "cil-arrow-right")
                contrib_col = _EMISSIVE_CAPTURE_TABLE.get(sp, (0.5, 0.5, 0.5))
                # Solid angle weight: larger / closer cells contribute more
                ox   = obbox.get("x", 0) + obbox.get("w", 80) / 2.0
                oy   = obbox.get("y", 0) + obbox.get("h", 50) / 2.0
                oz   = float(obbox.get("z", 3))
                dist = max(1.0, math.sqrt((cx-ox)**2 + (cy-oy)**2 + (cz-oz)**2*1e4))
                area = obbox.get("w", 80) * obbox.get("h", 50)
                w    = area / (dist * dist) * supersample
                contributors.append((contrib_col, w))

        if contributors:
            total_w = sum(w for _, w in contributors)
            avg_r   = sum(c[0]*w for c, w in contributors) / total_w
            avg_g   = sum(c[1]*w for c, w in contributors) / total_w
            avg_b   = sum(c[2]*w for c, w in contributors) / total_w
        else:
            # No contributors: sky colour from near-plane (ASTRO_CAPTURE_NEAR_PLANE)
            # Mirrors the CaptureSceneToScratchCubemap sky fallback
            avg_r, avg_g, avg_b = 0.55, 0.68, 0.82   # sky blue default

        face_base = (avg_r, avg_g, avg_b)

        # Build mip chain for this face
        face_dict: _PTDict[str, tuple] = {}
        for mip in range(_CAPTURE_NUM_MIPS):
            face_dict[f"mip_{mip}"] = gaussian_downsample_face_mip(
                face_base, mip, sigma_scale=0.7)
        capture.face_data[fi] = face_dict

    print(
        f"[ASTRO-CAPTURE] CaptureSceneToScratchCubemap: "
        f"cell_id={cell_id} slot={slot} "
        f"faces_captured={list(faces_to_capture)} "
        f"supersample={supersample}",
        file=sys.stderr,
    )
    return capture





@_ptdc
class AstroCellCaptureState:
    """
    Python equivalent of FCaptureComponentSceneState.

    Tracks the lifecycle of one reflection capture probe:
        cubemap_index: int     — slot in the global cubemap array
        fade_alpha:    float   — fade-in progress [0, 1]
        rendered_once: bool    — True once all 6 faces captured at least once
        is_dirty:      bool    — True when the capture needs a refresh
        cell_id:       str     — owning cell (Astro-specific field)

    鲁迅式：一个探针的一生——诞生于 FindOrAllocateCubemapIndex，
    成熟于所有面被捕获完毕，淡入于 fade_alpha 趋近 1.0，
    死亡于探针被 evict 或场景被清除。
    """
    cubemap_index: int   = -1
    fade_alpha:    float = 0.0
    rendered_once: bool  = False
    is_dirty:      bool  = True
    cell_id:       str   = ""
    # Per-face capture data: list of 6 face dicts, each with mip levels
    # Face dict: { "mip_0": (R,G,B), "mip_1": (R,G,B), ... }
    face_data: _PTList[_PTDict[str, tuple]] = _ptfield(
        default_factory=lambda: [{} for _ in range(_CAPTURE_NUM_FACES)])
    # Prefiltered specular: mip_level → average_colour (convolved)
    specular_prefilter: _PTDict[int, tuple] = _ptfield(default_factory=dict)
    # Diffuse SH irradiance (9 coefficients, 3 channels × 3 = 9 floats)
    diffuse_sh: _PTList[float] = _ptfield(default_factory=lambda: [0.0] * 9)


@_ptdc



def begin_capture_task(num_captures: int, reason: str = "UpdateCaptures") -> None:
    """
    Log the start of a reflection capture batch.
    Mirrors BeginReflectionCaptureSlowTask() — only logging in Astro (no UE slow task UI).
    """
    print(
        f"[ASTRO-CAPTURE] BeginReflectionCaptureSlowTask: "
        f"num={num_captures} reason={reason}",
        file=sys.stderr,
    )





def update_capture_task(capture_index: int, num_captures: int) -> None:
    """Mirrors UpdateReflectionCaptureSlowTask — progress logging."""
    if capture_index % max(1, num_captures // 5) == 0:
        pct = int(100.0 * capture_index / max(num_captures, 1))
        print(
            f"[ASTRO-CAPTURE] UpdateReflectionCaptureSlowTask: "
            f"{capture_index}/{num_captures} ({pct}%)",
            file=sys.stderr,
        )





def end_capture_task(num_captures: int) -> None:
    """Mirrors EndReflectionCaptureSlowTask."""
    print(
        f"[ASTRO-CAPTURE] EndReflectionCaptureSlowTask: "
        f"num={num_captures} complete",
        file=sys.stderr,
    )





def compute_capture_priority(
    cell_id:    str,
    bbox:       dict,
    camera_pos: tuple = (600.0, 450.0, 3.0),
) -> float:
    """
    Compute signed distance priority for runtime capture budget sorting.

    Mirrors ComputeRuntimeBudgetSignedDistance() from ReflectionEnvironmentCapture.cpp:
        For sphere probes: dist − InfluenceRadius
        For box probes:    Chebyshev distance to box surface

    2-D adaptation: uses 2-D bbox proximity to camera position.
    Lower = higher priority (closest captures rendered first).

    鲁迅式：优先级是稀缺资源分配的哲学——离得近的先照，离得远的等着。
    这不是歧视，是现实主义。
    """
    cx = bbox.get("x", 0) + bbox.get("w", 80) / 2.0
    cy = bbox.get("y", 0) + bbox.get("h", 50) / 2.0
    cz = float(bbox.get("z", 3))

    cam_x, cam_y, cam_z = camera_pos
    dx = cx - cam_x
    dy = cy - cam_y
    dz = (cz - cam_z) * 100.0   # scale z to world units

    influence_r = max(bbox.get("w", 80), bbox.get("h", 50)) / 2.0
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    return dist - influence_r   # negative = camera inside the probe influence





def convolve_capture_cubemap(cell_id: str) -> AstroCellCaptureState:
    """
    Convolve the captured cubemap to produce a pre-filtered specular environment.
    Mirrors the ConvolveCubeMap() pass called after CaptureSceneToScratchCubemap().

    For each mip level: average face_data across all 6 faces → convolve_specular_face
    → store result in capture.specular_prefilter[mip].

    Also computes diffuse irradiance SH from the mip-0 face data.

    鲁迅式：卷积是提炼——把六个方向的原始捕获数据，
    提炼成一份可以被任何粗糙度查询的预滤波环境贴图。
    这是从现象到本质的压缩，是科学的做法。
    """
    scene   = get_reflection_scene_data()
    capture = scene.allocated_captures.get(cell_id)
    if capture is None:
        return AstroCellCaptureState(cell_id=cell_id, cubemap_index=-1)

    for mip in range(_CAPTURE_NUM_MIPS):
        mip_key = f"mip_{mip}"
        face_cols = [
            capture.face_data[fi].get(mip_key, (0.5, 0.5, 0.5))
            for fi in range(_CAPTURE_NUM_FACES)
        ]
        avg_r = sum(c[0] for c in face_cols) / _CAPTURE_NUM_FACES
        avg_g = sum(c[1] for c in face_cols) / _CAPTURE_NUM_FACES
        avg_b = sum(c[2] for c in face_cols) / _CAPTURE_NUM_FACES
        face_avg = (avg_r, avg_g, avg_b)
        capture.specular_prefilter[mip] = convolve_specular_face(face_avg, mip)

    # Diffuse irradiance SH from mip-0 faces (highest resolution)
    mip0_faces = [
        capture.face_data[fi].get("mip_0", (0.5, 0.5, 0.5))
        for fi in range(_CAPTURE_NUM_FACES)
    ]
    capture.diffuse_sh = compute_diffuse_irradiance_sh(mip0_faces)

    capture.rendered_once = True
    capture.is_dirty      = False

    print(
        f"[ASTRO-CAPTURE] ConvolveCubeMap: "
        f"cell_id={cell_id} slot={capture.cubemap_index} "
        f"mips_computed={_CAPTURE_NUM_MIPS} "
        f"sh_L0=({capture.diffuse_sh[0]:.3f},{capture.diffuse_sh[1]:.3f},{capture.diffuse_sh[2]:.3f})",
        file=sys.stderr,
    )
    return capture





def update_reflection_captures(
    all_bboxes:    dict,
    camera_pos:    tuple = (600.0, 450.0, 3.0),
    timeslice:     bool  = True,
    force_all:     bool  = False,
) -> _PTList[str]:
    """
    Update dirty reflection captures in priority order.

    Mirrors the runtime reflection capture update loop in
    FScene::UpdateReflectionCaptureContents() / BeginRenderingReflectionCaptures():
      1. Collect dirty captures from allocated_captures
      2. Sort by compute_capture_priority (nearest first)
      3. Apply budget (ASTRO_CAPTURE_BUDGET = 0 → unlimited)
      4. For each: capture_scene_to_scratch_cubemap + convolve_capture_cubemap
      5. Fade-in (fade_alpha += dt / ASTRO_CAPTURE_FADE_TIME)

    Returns list of cell_ids updated this pass.

    鲁迅式：按距离排队，公平而现实——远处的探针等着，近处的先享受光照。
    这不是歧视，是优先级：视觉效果由近而远递减，计算预算由近而远递增。
    """
    scene   = get_reflection_scene_data()

    # Collect dirty + never-rendered captures from all_bboxes
    pending: _PTList[tuple] = []
    for cell_id, bbox in all_bboxes.items():
        cap = scene.allocated_captures.get(cell_id)
        if cap is None or cap.is_dirty or force_all:
            prio = compute_capture_priority(cell_id, bbox, camera_pos)
            pending.append((prio, cell_id, bbox))

    # Sort by priority (ascending distance → nearest first)
    pending.sort(key=lambda x: x[0])

    # Budget gate (0 = unlimited)
    budget    = ASTRO_CAPTURE_BUDGET if ASTRO_CAPTURE_BUDGET > 0 else len(pending)
    to_update = pending[:budget]

    if to_update:
        begin_capture_task(len(to_update), "UpdateReflectionCaptures")

    updated_cells: _PTList[str] = []
    dt = 1.0 / max(1, ASTRO_CAPTURE_TIMESLICE_FACES)  # fake dt per timeslice step

    for idx, (prio, cell_id, bbox) in enumerate(to_update):
        update_capture_task(idx, len(to_update))
        sp = all_bboxes[cell_id].get("species", "cil-arrow-right")

        if timeslice and not force_all:
            # Timesliced: capture ASTRO_CAPTURE_TIMESLICE_FACES per update
            for fi in range(ASTRO_CAPTURE_TIMESLICE_FACES):
                capture_scene_to_scratch_cubemap(
                    cell_id, bbox, sp, all_bboxes, face_index=fi)
        else:
            # Full capture: all 6 faces at once (mirrors "fast render on load")
            capture_scene_to_scratch_cubemap(
                cell_id, bbox, sp, all_bboxes, face_index=-1)

        convolve_capture_cubemap(cell_id)

        # Fade-in update: increment fade_alpha toward 1.0
        cap = scene.allocated_captures.get(cell_id)
        if cap:
            cap.fade_alpha = min(1.0, cap.fade_alpha + dt / max(ASTRO_CAPTURE_FADE_TIME, 1e-3))

        updated_cells.append(cell_id)

    if to_update:
        end_capture_task(len(to_update))

    return updated_cells





@_ptdc
class AstroCellRealTimeSkyCapture:
    """
    Python equivalent of FRealTimeSlicedReflectionCapture.

    State machine for the timesliced sky capture update:
        current_face:  int  — which cube face is being rendered this frame (0..5)
        faces_done:    int  — bitmask of completed faces (all done when == 0x3F)
        cube_size:     int  — current cube resolution (overrideable)
        is_valid:      bool — True after at least one complete convolution
        invalidated:   bool — set True when sky conditions change (forces re-capture)

    The C++ FRealTimeSlicedReflectionCapture holds similar state in the scene:
        ConvolvedSkyRenderTarget[2] → convolve buffers (inner/outer mip chains)
        bConvolvedSkyRenderTargetInvalid → invalidated flag

    鲁迅式：分时渲染是以时间换空间的妥协——
    每帧只画一面，六帧后才是完整的天空；
    不完整的天空已经够用了，这就是实时的现实。
    """
    current_face:    int   = 0
    faces_done:      int   = 0       # bitmask 0b000000 .. 0b111111
    cube_size:       int   = ASTRO_RT_DEFAULT_CUBE_SIZE
    is_valid:        bool  = False
    invalidated:     bool  = True
    frame_count:     int   = 0
    # Sky face radiance: face_index → (R, G, B)
    sky_face_radiance: _PTList[tuple] = _ptfield(
        default_factory=lambda: [(0.55, 0.68, 0.82)] * _CAPTURE_NUM_FACES)
    # Cloud face radiance (low-resolution): face_index → (R, G, B)
    cloud_face_radiance: _PTList[tuple] = _ptfield(
        default_factory=lambda: [(0.85, 0.88, 0.92)] * _CAPTURE_NUM_FACES)
    # Convolve outputs (pre-filtered specular mip chain)
    convolve_specular: _PTDict[int, tuple] = _ptfield(default_factory=dict)
    diffuse_sh:        _PTList[float]      = _ptfield(default_factory=lambda: [0.0]*9)


# Module-level real-time sky capture state singleton
_ASTRO_RT_SKY_CAPTURE: AstroCellRealTimeSkyCapture = AstroCellRealTimeSkyCapture()





def get_rt_sky_capture() -> AstroCellRealTimeSkyCapture:
    """Return the module-level AstroCellRealTimeSkyCapture singleton."""
    return _ASTRO_RT_SKY_CAPTURE





def validate_sky_light_rt_capture(
    has_sky_mesh: bool,
    sky_material_changed: bool,
    is_being_edited: bool = False,
) -> None:
    """
    Validate / invalidate the real-time sky capture state.
    Mirrors FScene::ValidateSkyLightRealTimeCapture():
        If sky conditions changed (sky mesh added/removed, material changed),
        set bConvolvedSkyRenderTargetInvalid = True to force a full re-capture.

    鲁迅式：验证是诚实的代价——每次场景发生变化，
    旧的天空就失效了，必须诚实地重新捕获。
    假装旧的还有效，是偷懒，也是错误。
    """
    capture = get_rt_sky_capture()
    if sky_material_changed or has_sky_mesh != capture.is_valid or is_being_edited:
        capture.invalidated = True
        capture.faces_done  = 0
        capture.current_face = 0
        print(
            f"[ASTRO-RT-CAPTURE] ValidateSkyLightRealTimeCapture — invalidated: "
            f"has_sky_mesh={has_sky_mesh} "
            f"sky_changed={sky_material_changed} "
            f"editing={is_being_edited}",
            file=sys.stderr,
        )





def render_sky_pass_for_capture(
    face_index:       int,
    atmosphere_color: tuple = (0.55, 0.68, 0.82),
    cloud_color:      tuple = (0.85, 0.88, 0.92),
    sun_direction:    tuple = (0.3, -0.8, 0.5),
    include_clouds:   bool  = True,
    depth_buffer:     bool  = ASTRO_RT_DEPTH_BUFFER,
) -> tuple:
    """
    Render one sky face for the real-time sky env map capture.

    Mirrors RenderSkyPassForCapture() + the per-face render loop in
    UpdateSkyEnvMap():
      - Sky atmosphere scatter (SkyAtmosphereRendering.cpp port → analytic approx)
      - Volumetric cloud compositing (low-res, CVarRealTimeReflectionCaptureVolumetricCloudResolutionDivider)
      - Fog contribution (FogRendering.cpp → distance-weighted blend)
      - Optional shadow from opaque (ASTRO_RT_SHADOW_FROM_OPAQUE)
      - Optional depth buffer (ASTRO_RT_DEPTH_BUFFER → height-fog attenuation)

    Returns (R, G, B) sky radiance for the given face direction.

    鲁迅式：天空是每帧都在变化的背景——太阳西沉，云彩移动，
    大气散射随角度而改变。我们无法一劳永逸地捕获它，
    只能帧帧跟进，面面不落。
    """
    # Face normal directions
    normals = [
        ( 1, 0, 0), (-1, 0, 0),
        ( 0, 1, 0), ( 0,-1, 0),
        ( 0, 0, 1), ( 0, 0,-1),
    ]
    nx, ny, nz = normals[face_index % _CAPTURE_NUM_FACES]

    # ── Atmosphere scatter (Rayleigh + Mie analytic approximation) ───────
    # Sun angle relative to this face normal
    sx, sy, sz = sun_direction
    s_len = math.sqrt(sx*sx + sy*sy + sz*sz)
    if s_len > 1e-6:
        sx, sy, sz = sx/s_len, sy/s_len, sz/s_len
    cos_sun = max(0.0, nx*sx + ny*sy + nz*sz)

    # Rayleigh scatter: blue sky dominates at angles away from sun
    rayleigh  = 1.0 - cos_sun * 0.6
    sky_r = atmosphere_color[0] * rayleigh + cos_sun * 0.95
    sky_g = atmosphere_color[1] * rayleigh + cos_sun * 0.85
    sky_b = atmosphere_color[2] * rayleigh + cos_sun * 0.70

    # ── Cloud compositing (low-res, ASTRO_RT_CLOUD_RES_DIVIDER) ──────────
    if include_clouds:
        cloud_weight = max(0.0, cloud_color[0] + cloud_color[1] + cloud_color[2]) / 3.0
        cloud_frac   = min(0.35, cloud_weight * 0.4) / max(ASTRO_RT_CLOUD_RES_DIVIDER, 1)
        sky_r = sky_r * (1.0 - cloud_frac) + cloud_color[0] * cloud_frac
        sky_g = sky_g * (1.0 - cloud_frac) + cloud_color[1] * cloud_frac
        sky_b = sky_b * (1.0 - cloud_frac) + cloud_color[2] * cloud_frac

    # ── Depth buffer height fog (ASTRO_RT_DEPTH_BUFFER attenuation) ───────
    if depth_buffer:
        # Height-based fog: lower faces (face 3 = -Y) get more fog
        fog_factor = max(0.0, -ny) * 0.15
        fog_r, fog_g, fog_b = 0.8, 0.85, 0.9   # fog colour
        sky_r = sky_r * (1-fog_factor) + fog_r * fog_factor
        sky_g = sky_g * (1-fog_factor) + fog_g * fog_factor
        sky_b = sky_b * (1-fog_factor) + fog_b * fog_factor

    # ── Shadow from opaque (optional, ASTRO_RT_SHADOW_FROM_OPAQUE) ────────
    if ASTRO_RT_SHADOW_FROM_OPAQUE:
        # Darken the +Y face (sun-facing) slightly for opaque-mesh shadow
        shadow_mult = 1.0 - max(0.0, ny) * 0.12
        sky_r *= shadow_mult
        sky_g *= shadow_mult
        sky_b *= shadow_mult

    return (max(0.0, sky_r), max(0.0, sky_g), max(0.0, sky_b))





def is_reflection_capture_available() -> bool:
    """True when baked captures are allowed (IsStaticLightingAllowed — always True here)."""
    return True





def compute_capture_fade(
    fade_start_value: float,
    fade_target_value: float,
    fade_start_time:   float,
    current_time:      float,
    duration:          float,
) -> float:
    """
    Lerp fade for reflection capture transitions.
    Port of FCaptureComponentSceneState::ComputeCurrentFade().

    鲁迅式：淡入淡出是过渡期的妥协——突变是震惊，渐变是说服。
    """
    if fade_start_value == fade_target_value:
        return fade_target_value
    if duration <= 0.0:
        return fade_target_value
    t = max(0.0, min(1.0, (current_time - fade_start_time) / duration))
    return fade_start_value + (fade_target_value - fade_start_value) * t


