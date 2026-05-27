"""server_layered_routes.py — The invoke_last_filter.

CCCL f984c90's public API is simple:

    CUB_RUNTIME_FUNCTION static cudaError_t TopK(
        void* d_temp_storage,
        KeyInputIteratorT d_keys_in,
        KeyOutputIteratorT d_keys_out, ...) {
        return dispatch(...);   // delegates to the internal pipeline
    }

The public function does nothing except validate inputs and delegate
to dispatch().  That's the pattern for good API design: thin shell,
thick engine.

Our server routes follow the same pattern:
  1. Validate the HTTP request
  2. Delegate to generate_layered_topology() or regenerate_single_region()
  3. Format the response

The routes are registered via register_layered_routes(app), called
from server.py — like CCCL's DeviceTopK::Dispatch being instantiated
inside the TopK static method.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Request / Response models — the API contract
# ═══════════════════════════════════════════════════════════════════════════

class LayeredTopologyRequest(BaseModel):
    """Request for the layered topology pipeline.

    Like CCCL's DeviceTopK parameters: d_keys_in (text),
    k (max_regions), d_temp_storage size (canvas dimensions).
    """
    text: str = Field(..., min_length=1, description="Full system description text")
    canvas_width: Optional[int] = Field(None, ge=400, le=4000)
    canvas_height: Optional[int] = Field(None, ge=300, le=3000)
    max_regions: int = Field(8, ge=1, le=12)
    model: str = Field("", description="LLM model override")
    output_format: str = Field("elk", description="elk | mastergo | both")
    skip_refinement: bool = Field(False)


class RegionRegenerateRequest(BaseModel):
    """Request to regenerate a single region.

    Like re-running one pass of CCCL's radix loop without
    restarting the entire pipeline.  The DoubleBuffer pattern
    means we swap in a new result for one region and re-merge.
    """
    text: str = Field(..., min_length=1)
    region_id: str = Field(..., min_length=1)
    # The client sends back the full pipeline state so we can
    # recompose without re-running everything
    regions: list = Field(..., description="Region plans from previous run")
    canvas: dict = Field(..., description="Previous canvas state")
    model: str = Field("")


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Route registration — the DeviceTopK::TopK() wrapper
# ═══════════════════════════════════════════════════════════════════════════

def register_layered_routes(app: FastAPI) -> None:
    """Register the layered topology endpoints.

    Called from server.py during app initialization.
    Like CCCL's template instantiation: the routes are generic
    functions that get bound to a specific FastAPI app instance.
    """

    @app.post("/api/topology-layered")
    async def api_topology_layered(request_data: LayeredTopologyRequest) -> JSONResponse:
        """Layered multi-pass topology pipeline.

        Full dispatch(): intent → region plan → per-region gen → compose.

        Like CCCL's DeviceTopK::TopK():
          1. Validate inputs (this function)
          2. Delegate to dispatch() (generate_layered_topology)
          3. Format output (JSONResponse)

        Request:
            text: Full system description (NEVER truncated)
            canvas_width: Optional canvas width override
            canvas_height: Optional canvas height override
            max_regions: Maximum regions to plan (1-12, default 8)
            model: LLM model override
            output_format: "elk" | "mastergo" | "both"
            skip_refinement: Skip optional refinement pass

        Response:
            success: bool
            canvas: {width, height}
            elk: Unified ELK graph (all regions composed)
            layers: Layer metadata (name, z_index, visible, locked)
            regions: Region plans with bboxes
            cross_region_edges: Edges crossing region boundaries
            intent: Parsed user intent summary
            diagnostics: Per-stage timing and stats
            elapsed_ms: Total pipeline time
        """
        try:
            # Lazy imports — like CCCL's template instantiation at call time
            from backend.pipeline.topology.layered_pipeline import (
                generate_layered_topology,
            )
            from backend.config import get_settings
            from backend.ai_engine import AIEngine

            ai_engine = AIEngine(get_settings())

            # ── Dispatch to the pipeline ──
            result = await generate_layered_topology(
                text=request_data.text,
                ai_engine=ai_engine,
                model=request_data.model,
                canvas_width=request_data.canvas_width,
                canvas_height=request_data.canvas_height,
                max_regions=request_data.max_regions,
                skip_refinement=request_data.skip_refinement,
            )

            response = result.to_dict()

            # Optional mastergo conversion (M14: prefer layered converter)
            if request_data.output_format in ("mastergo", "both") and result.canvas:
                try:
                    from backend.pipeline.topology.mastergo_schema import (
                        layered_to_mastergo_layout,
                        elk_to_mastergo_layout,
                    )
                    # M14: use region-aware converter when regions are available
                    if result.regions:
                        mastergo = layered_to_mastergo_layout(result)
                        response["mastergo"] = mastergo.to_layered_dict()
                        response["mastergo_import"] = mastergo.to_mastergo_import()
                    else:
                        # Fallback to flat ELK converter
                        elk = result.canvas.elk_graph
                        w = result.canvas.width
                        h = result.canvas.height
                        mastergo = elk_to_mastergo_layout(elk, w, h)
                        response["mastergo"] = mastergo.to_list()
                    response["mastergo_stats"] = mastergo.stats()
                except Exception as e:
                    response["mastergo_error"] = str(e)

            return JSONResponse(response)

        except Exception as e:
            logger.exception("api_topology_layered error")
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=500,
            )

    @app.post("/api/region-regenerate")
    async def api_region_regenerate(request_data: RegionRegenerateRequest) -> JSONResponse:
        """Regenerate a single region without re-running the full pipeline.

        Like re-running one pass of CCCL's radix loop:
          - The DoubleBuffer swaps in the new subgraph
          - The compositor re-merges with the updated region
          - Other regions are untouched

        This is the "regenerate this region" button in the frontend.
        The client sends back the full pipeline state (regions + canvas)
        so we can recompose without re-running region planning.
        """
        try:
            from backend.pipeline.topology.layered_pipeline import (
                regenerate_single_region,
            )
            from backend.pipeline.topology.user_intent_parser import (
                parse_user_intent,
            )
            from backend.pipeline.topology.region_planner import PlannedRegion
            from backend.pipeline.topology.canvas_compositor import (
                ComposedCanvas,
            )
            from backend.config import get_settings
            from backend.ai_engine import AIEngine

            ai_engine = AIEngine(get_settings())

            # Reconstruct pipeline state from client data
            intent = parse_user_intent(request_data.text)
            regions = [
                PlannedRegion(
                    id=r.get("id", f"region_{i}"),
                    name=r.get("name", f"Region {i}"),
                    bbox=r.get("bbox", {"x": 0, "y": 0, "width": 400, "height": 300}),
                    description=r.get("description", ""),
                    entity_hints=r.get("entity_hints", []),
                    style=r.get("style", {}),
                    priority=i,
                )
                for i, r in enumerate(request_data.regions)
            ]

            canvas_data = request_data.canvas
            canvas = ComposedCanvas(
                width=canvas_data.get("width", 900),
                height=canvas_data.get("height", 500),
                elk_graph=canvas_data.get("elk", {}),
            )

            result = await regenerate_single_region(
                region_id=request_data.region_id,
                text=request_data.text,
                intent=intent,
                regions=regions,
                canvas=canvas,
                ai_engine=ai_engine,
                model=request_data.model,
            )

            return JSONResponse(result.to_dict())

        except Exception as e:
            logger.exception("api_region_regenerate error")
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=500,
            )

    # ── M14: Dedicated MasterGo export endpoint ──

    class MastergoExportRequest(BaseModel):
        """Request to export an existing topology result as MasterGo format.

        The client sends back a previous pipeline result (elk + regions)
        and gets back the MasterGo Import API compatible JSON.
        This avoids re-running the LLM pipeline just to change export format.
        """
        elk: dict = Field(..., description="ELK graph from previous pipeline run")
        regions: list = Field(default=[], description="Region plans from previous run")
        canvas_width: int = Field(900, ge=400, le=4000)
        canvas_height: int = Field(500, ge=300, le=3000)
        format: str = Field("import", description="'import' for MasterGo API, 'flat' for element list, 'layered' for full")

    @app.post("/api/mastergo-export")
    async def api_mastergo_export(request_data: MastergoExportRequest) -> JSONResponse:
        """Convert an existing topology to MasterGo export format.

        Like CCCL's final output formatting: the computation is done,
        we're just reshaping the result for a different consumer (MasterGo
        instead of the web renderer).

        Supports three output formats:
          - 'import': MasterGo Import API document structure (frames/groups/rectangles)
          - 'layered': Full layout with layer metadata and element-to-layer mapping
          - 'flat': Simple element list (legacy mastergo_all_layoutobj.txt format)
        """
        try:
            from backend.pipeline.topology.mastergo_schema import (
                elk_to_mastergo_layout,
                layered_to_mastergo_layout,
                MastergoLayout,
                BBox,
            )

            if request_data.regions:
                # Build a lightweight result-like object for the layered converter
                from types import SimpleNamespace

                # Reconstruct canvas
                canvas = SimpleNamespace(
                    width=request_data.canvas_width,
                    height=request_data.canvas_height,
                    elk_graph=request_data.elk,
                    cross_region_edges=[],
                )

                # Reconstruct regions as namespace objects
                regions = []
                for i, r in enumerate(request_data.regions):
                    regions.append(SimpleNamespace(
                        id=r.get("id", f"region_{i}"),
                        name=r.get("name", f"Region {i+1}"),
                        bbox=r.get("bbox", {"x": 0, "y": 0, "width": 200, "height": 200}),
                    ))

                fake_result = SimpleNamespace(
                    canvas=canvas,
                    regions=regions,
                    intent=None,
                )

                mastergo = layered_to_mastergo_layout(fake_result)
            else:
                mastergo = elk_to_mastergo_layout(
                    request_data.elk,
                    request_data.canvas_width,
                    request_data.canvas_height,
                )

            # Choose output format
            fmt = request_data.format
            if fmt == "import":
                data = mastergo.to_mastergo_import()
            elif fmt == "layered":
                data = mastergo.to_layered_dict()
            else:
                data = {"elements": mastergo.to_list(), "edges": mastergo.edges}

            return JSONResponse({
                "success": True,
                "format": fmt,
                "data": data,
                "stats": mastergo.stats(),
            })

        except Exception as e:
            logger.exception("api_mastergo_export error")
            return JSONResponse(
                {"success": False, "error": str(e)},
                status_code=500,
            )
