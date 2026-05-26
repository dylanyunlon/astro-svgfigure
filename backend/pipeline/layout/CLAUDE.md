# Layout Detection Pipeline

## Module Map
```
detect.py       VisionDetectConfig, vision_detect, prompts, fallback CCL
elk_bridge.py   elk_to_mastergo, elk_extract_edges (full geometry preserved)
refine.py       refine_single_region, iterative_refine (crop-and-re-detect)
merge.py        merge_multi_state (Hungarian+UF), infer_hidden, grid_snap
stage.py        stage_omniparser_detect, handler, availability check
```

## Priority Chain (Stage 0.5)
1. elk_graph present → elk_to_mastergo() — 0 API calls, designer precision
2. elk_graph absent → vision_detect() — 1 vision API call per frame
3. No API → _fallback_detect() — scipy CCL

## Data Format (mastergo-compatible)
```json
[{"id": "nav", "name": "导航", "bbox": {"x": 13, "y": 185, "width": 52, "height": 52}}]
```
