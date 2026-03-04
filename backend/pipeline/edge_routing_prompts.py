"""
Advanced Edge Routing Prompts
Neural-network level multi-layer nested arrow system prompts.
"""

ADVANCED_EDGE_ROUTING_SYSTEM_PROMPT = """

=== ADVANCED EDGE ROUTING (Neural-Network Level) ===

You MUST use the "advanced" field on edges to express complex arrow semantics.
Each edge can have an "advanced" object with these fields:

1. semanticType (REQUIRED for non-trivial edges):
   - "data_flow": Standard forward data flow (solid arrow)
   - "gradient_flow": Gradient backpropagation (dashed, red-ish)
   - "skip_connection": ResNet-style skip/residual connection (curved arc, green)
   - "optional_path": Optional/conditional path (dashed, gray)
   - "inference_only": Used only during inference (dash-dot, purple)
   - "fan_out": One output splits to multiple modules
   - "fan_in": Multiple modules converge to one node
   - "feedback": Feedback/recurrent connection (dashed curve, purple)
   - "attention": Attention mechanism connection (dotted, red)
   - "concatenation": Tensor concatenation (solid, teal)
   - "residual": Residual addition (curved, green)
   - "cross_boundary": Edge crossing group boundaries

2. lineStyle: "solid" | "dashed" | "dotted" | "double"
3. strokeDasharray: SVG dash pattern, e.g. "5,5" or "10,3,3,3"
4. directionality: "directed" | "bidirectional" | "undirected"
5. edgeLabels: [{text, position (0-1), fontSize}]
6. crossesGroupBoundary: true for hierarchy edges
7. routing: "ORTHOGONAL" | "SPLINES" | "POLYLINE"
8. curvature: 0.0-1.0 for spline routing
9. priority: Higher = better routing
10. sourceArrow / targetArrow: "arrow"|"none"|"diamond"|"circle"|"open"
11. strokeWidth: default 1.5
12. strokeColor: Hex color

=== EDGE EXAMPLES ===

Skip Connection:
{"id":"skip_1","sources":["input"],"targets":["add"],
 "advanced":{"semanticType":"skip_connection","routing":"SPLINES",
  "curvature":0.7,"strokeWidth":2,"strokeColor":"#4CAF50"}}

Gradient Flow:
{"id":"grad_1","sources":["loss"],"targets":["encoder"],
 "advanced":{"semanticType":"gradient_flow","lineStyle":"dashed",
  "strokeDasharray":"8,4","strokeColor":"#E57373",
  "edgeLabels":[{"text":"grad","position":0.5}]}}

Labeled Edge:
{"id":"e_latent","sources":["encoder"],"targets":["decoder"],
 "advanced":{"semanticType":"data_flow",
  "edgeLabels":[{"text":"z ~ N(0,1)","position":0.5}]}}

Bidirectional:
{"id":"bidir_1","sources":["a"],"targets":["b"],
 "advanced":{"directionality":"bidirectional",
  "sourceArrow":"arrow","targetArrow":"arrow","strokeWidth":2}}

Cross-boundary:
{"id":"cross_1","sources":["group_a.enc"],"targets":["group_b.dec"],
 "advanced":{"semanticType":"cross_boundary",
  "crossesGroupBoundary":true,"routing":"ORTHOGONAL"}}

=== COMPOUND NODES ===
Use nested children for groups.
Set advanced.crossesGroupBoundary=true for cross-group edges.

=== RULES ===
- EVERY non-trivial edge MUST have "advanced" field
- Use ORTHOGONAL for clean diagrams, SPLINES for skip connections
"""


GROK_EDGE_ROUTING_SYSTEM_ADDON = """

=== CRITICAL: Advanced Arrow Rendering Instructions ===

For EACH arrow type, describe precisely:
1. Orthogonal: right-angle bends, clean routing (DEFAULT for architecture diagrams)
2. Fan-out: arrows diverge from single source symmetrically
3. Fan-in: arrows converge to single target
4. Dashed: indicating gradient/optional/inference paths
5. Bidirectional: double-headed arrows
6. Curved/Spline: smooth arcs for skip connections and residual paths
7. Labeled: text at midpoint with white background
8. Cross-boundary: arrows crossing group borders (parent→grandchild, etc.)
9. Bent/Elbow: right-angle bent arrows for routing around sibling nodes

For EACH arrow specify: start/end module, exit/entry direction,
bends, line style, color, label, curvature.

=== HIERARCHICAL GROUPING (Architecture Diagrams) ===

When generating points for architecture diagrams:
- Parent-child: parent node visually CONTAINS children as a group
- Sibling grouping: children of the same parent share a BORDERLESS,
  semi-transparent background region. NO hard borders on the group —
  only a subtle tint (e.g., rgba(100,150,255,0.08)) to show they're related
- Grandchild nesting: each deeper level gets a slightly different tint
  to create visual depth hierarchy
- Neural-network level nesting: describe EACH layer explicitly with its
  own background tint and contained elements
- Group labels appear at the TOP of the borderless background region

=== DYNAMIC VISUAL ELEMENTS ===

For icons, illustrations, avatars, and visual elements inside nodes:
- ALWAYS describe what the visual should depict using natural language
  (e.g., "a small detailed illustration of a microscope in flat vector style")
- NEVER use emoji, Unicode symbols, or hardcoded character references
- The image model (Gemini 3 Pro Image) will generate these visuals natively
  from the text description in the prompt
- Each node that needs an icon should specify: what it depicts, its style
  (flat, line-art, isometric), its approximate size relative to the node,
  and its position within the node (top-left, center, etc.)
"""


def get_topology_prompt_with_edge_routing(base_prompt):
    return base_prompt + ADVANCED_EDGE_ROUTING_SYSTEM_PROMPT


def get_grok_prompt_with_edge_routing(base_prompt):
    return base_prompt + GROK_EDGE_ROUTING_SYSTEM_ADDON