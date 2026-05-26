# Topology Generation Pipeline

Single LLM call produces ~8 nodes. Actual figure needs ~50+.
Fix: 5-step pipeline with schema validation gates.

Step 1: entity_extractor    -> 30+ entities (reject if <20)
Step 2: relationship_mapper -> edges (reject if <15)
Step 3: hierarchy_builder   -> groups (every entity assigned)
Step 4: icon_classifier     -> iconHints (rule-based, no API)
Step 5: elk_assembler       -> complete ELK JSON

Each step retries up to 3x. Never silently drops content.
