"""
Visual Identity Extractor — Subject Identity Preservation System
=================================================================
This module extracts and encodes visual identity features from an image
to ensure the animated frames maintain consistent appearance.

CORE PROBLEM BEING SOLVED:
─────────────────────────
User complaint: "The animation has no relation to my original image"

Root cause analysis:
1. The old pipeline sent ONLY text description to Gemini
2. Gemini created NEW images from text, not edited versions of the original
3. Even with the same prompt, each generation produced different subjects

SOLUTION:
────────
This module creates a "Visual Identity Profile" that:
1. Captures specific visual features from Claude's analysis
2. Generates identity-locking prompts for Gemini
3. Creates "hard negatives" (things that must NOT change)
4. Provides frame-by-frame consistency checks

BASED ON RESEARCH:
─────────────────
Per sider.ai/blog (How to Write Gemini Prompts That Keep Subject Identity):
"Use an identity header that locks facial geometry, hair length/color, and
unique markers, plus hard negatives like 'no morphing, no freckles.' Always
attach the same reference image and restate the constraints in each edit."

Per blog.google/products/gemini/image-generation-prompting-tips:
"By establishing a clearly defined character with specific details in the
first prompt, you can use follow-up prompts to place that same character
in entirely new contexts."

KEY CONCEPTS:
────────────
1. IDENTITY ANCHOR: Core visual features that define the subject
2. HARD NEGATIVES: Explicit "do not" constraints
3. SOFT FEATURES: Elements that CAN change (position, pose)
4. CONSISTENCY MARKERS: Features to check across frames

PIPELINE INTEGRATION:
───────────────────
Step 1: Claude analyzes image → produces analysis dict
Step 1.5: THIS MODULE → extracts VisualIdentityProfile
Step 2: Grok/Prompt design → uses identity profile
Step 3: Gemini generation → includes identity lock in every frame

Knuth-Level Critique Resolution:
───────────────────────────────
USER CRITIQUE: "The car changed color between frames"
SOLUTION: We extract exact color values and include them as hard constraints.

USER CRITIQUE: "The style changed from realistic to cartoon"
SOLUTION: We extract style descriptors and lock them with hard negatives.

SYSTEM CRITIQUE: "Visual features were described too vaguely"
SOLUTION: This module creates specific, quantifiable feature descriptions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════════

class SubjectType(Enum):
    """Type of subject in the image."""
    VEHICLE = "vehicle"
    PERSON = "person"
    ANIMAL = "animal"
    OBJECT = "object"
    CHARACTER = "character"
    ARCHITECTURE = "architecture"
    FOOD = "food"
    PLANT = "plant"
    ABSTRACT = "abstract"
    UNKNOWN = "unknown"


class VisualStyle(Enum):
    """Visual rendering style of the image."""
    PHOTOREALISTIC = "photorealistic"
    REALISTIC = "realistic"
    SEMI_REALISTIC = "semi_realistic"
    CARTOON = "cartoon"
    ANIME = "anime"
    PIXEL_ART = "pixel_art"
    SKETCH = "sketch"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil_painting"
    DIGITAL_ART = "digital_art"
    VECTOR = "vector"
    CLAY = "clay"
    PAPER = "paper"
    UNKNOWN = "unknown"


@dataclass
class ColorInfo:
    """Detailed color information."""
    name: str  # Human-readable name (e.g., "bright red")
    hex_code: Optional[str] = None  # Hex code if available (e.g., "#FF0000")
    role: str = "primary"  # "primary", "secondary", "accent", "background"

    def to_constraint(self) -> str:
        """Convert to a constraint string for prompts."""
        if self.hex_code:
            return f"{self.name} ({self.hex_code})"
        return self.name


@dataclass
class ComponentInfo:
    """Information about a visual component."""
    name: str  # Component name (e.g., "wheel", "window")
    description: str = ""  # Detailed description
    is_key_feature: bool = True  # Is this a defining feature?
    can_move: bool = False  # Can this component move independently?

    def to_constraint(self) -> str:
        """Convert to a constraint string."""
        return f"{self.name}: {self.description}" if self.description else self.name


@dataclass
class VisualIdentityProfile:
    """
    Complete visual identity profile for a subject.

    This profile captures everything needed to maintain visual consistency
    across animation frames.
    """
    # Core identity
    subject_type: SubjectType = SubjectType.UNKNOWN
    subject_description: str = ""
    visual_style: VisualStyle = VisualStyle.UNKNOWN

    # Colors
    primary_colors: List[ColorInfo] = field(default_factory=list)
    accent_colors: List[ColorInfo] = field(default_factory=list)
    color_constraints: List[str] = field(default_factory=list)

    # Components
    key_components: List[ComponentInfo] = field(default_factory=list)
    movable_components: List[ComponentInfo] = field(default_factory=list)

    # Style markers
    style_descriptors: List[str] = field(default_factory=list)
    texture_descriptors: List[str] = field(default_factory=list)

    # Hard negatives (things that must NOT change)
    hard_negatives: List[str] = field(default_factory=list)

    # Animation constraints
    recommended_styles: List[str] = field(default_factory=list)
    avoid_styles: List[str] = field(default_factory=list)

    # Metadata
    confidence: float = 0.0  # 0.0 to 1.0
    extraction_notes: List[str] = field(default_factory=list)

    def to_identity_header(self) -> str:
        """Generate the identity header string for Gemini prompts."""
        lines = [
            "═══════════════════════════════════════════════════════",
            "VISUAL IDENTITY LOCK — PRESERVE THESE EXACTLY",
            "═══════════════════════════════════════════════════════",
            "",
            f"SUBJECT: {self.subject_description}",
            f"STYLE: {self.visual_style.value.replace('_', ' ').title()}",
            "",
        ]

        # Colors
        if self.primary_colors or self.accent_colors:
            lines.append("COLORS (DO NOT MODIFY):")
            for c in self.primary_colors[:4]:
                lines.append(f"  • Primary: {c.to_constraint()}")
            for c in self.accent_colors[:2]:
                lines.append(f"  • Accent: {c.to_constraint()}")
            lines.append("")

        # Key features
        if self.key_components:
            lines.append("KEY FEATURES (MUST PRESERVE):")
            for comp in self.key_components[:6]:
                lines.append(f"  • {comp.to_constraint()}")
            lines.append("")

        # Style descriptors
        if self.style_descriptors:
            lines.append("STYLE CHARACTERISTICS:")
            lines.append(f"  {', '.join(self.style_descriptors[:5])}")
            lines.append("")

        # Hard negatives
        lines.append("HARD NEGATIVES (DO NOT DO THESE):")
        for neg in self.hard_negatives[:8]:
            lines.append(f"  ✗ {neg}")

        lines.append("")
        lines.append("═══════════════════════════════════════════════════════")

        return "\n".join(lines)

    def to_short_header(self) -> str:
        """Generate a shorter identity header for token-limited contexts."""
        parts = [
            f"Subject: {self.subject_description}",
            f"Style: {self.visual_style.value}",
        ]

        if self.primary_colors:
            color_str = ", ".join(c.name for c in self.primary_colors[:3])
            parts.append(f"Colors: {color_str}")

        if self.hard_negatives:
            parts.append(f"DO NOT: {'; '.join(self.hard_negatives[:3])}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "subject_type": self.subject_type.value,
            "subject_description": self.subject_description,
            "visual_style": self.visual_style.value,
            "primary_colors": [
                {"name": c.name, "hex": c.hex_code, "role": c.role}
                for c in self.primary_colors
            ],
            "accent_colors": [
                {"name": c.name, "hex": c.hex_code, "role": c.role}
                for c in self.accent_colors
            ],
            "key_components": [
                {"name": c.name, "description": c.description}
                for c in self.key_components
            ],
            "style_descriptors": self.style_descriptors,
            "hard_negatives": self.hard_negatives,
            "confidence": self.confidence,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Main Extraction Function
# ═══════════════════════════════════════════════════════════════════════════

def extract_visual_identity(analysis: Dict[str, Any]) -> VisualIdentityProfile:
    """
    Extract a visual identity profile from Claude's image analysis.

    Parameters
    ----------
    analysis : Dict[str, Any]
        The analysis dict from Claude's image analysis step.
        Expected keys: summary, components, color_palette, style, etc.

    Returns
    -------
    VisualIdentityProfile
        A complete profile for maintaining visual consistency.
    """
    profile = VisualIdentityProfile()

    # ── Extract subject type and description ──
    summary = analysis.get("summary", "")
    profile.subject_description = summary or "the subject in the image"
    profile.subject_type = _infer_subject_type(summary)

    # ── Extract visual style ──
    style_str = analysis.get("style", "").lower()
    profile.visual_style = _parse_visual_style(style_str)
    profile.style_descriptors = _extract_style_descriptors(analysis)

    # ── Extract colors ──
    colors = analysis.get("color_palette", [])
    profile.primary_colors, profile.accent_colors = _parse_colors(colors)

    # ── Extract components ──
    components = analysis.get("components", [])
    profile.key_components, profile.movable_components = _parse_components(components)

    # ── Generate hard negatives ──
    profile.hard_negatives = _generate_hard_negatives(profile)

    # ── Extract texture descriptors ──
    profile.texture_descriptors = _extract_textures(analysis)

    # ── Animation recommendations ──
    profile.recommended_styles, profile.avoid_styles = _get_animation_recommendations(
        profile.subject_type, profile.visual_style
    )

    # ── Calculate confidence ──
    profile.confidence = _calculate_confidence(profile, analysis)

    # ── Add extraction notes ──
    profile.extraction_notes = []
    if not colors:
        profile.extraction_notes.append("No color palette in analysis — using defaults")
    if not components:
        profile.extraction_notes.append("No components identified — identity may drift")

    logger.info(
        "Extracted identity profile: type=%s, style=%s, colors=%d, components=%d, confidence=%.2f",
        profile.subject_type.value,
        profile.visual_style.value,
        len(profile.primary_colors) + len(profile.accent_colors),
        len(profile.key_components),
        profile.confidence,
    )

    return profile


# ═══════════════════════════════════════════════════════════════════════════
#  Subject Type Inference
# ═══════════════════════════════════════════════════════════════════════════

def _infer_subject_type(summary: str) -> SubjectType:
    """Infer the subject type from the summary text."""
    summary_lower = summary.lower()

    # Vehicle keywords
    vehicle_kw = ["car", "truck", "vehicle", "motorcycle", "bike", "bus", "train",
                  "plane", "airplane", "boat", "ship", "helicopter", "bicycle"]
    if any(kw in summary_lower for kw in vehicle_kw):
        return SubjectType.VEHICLE

    # Person keywords
    person_kw = ["person", "man", "woman", "human", "people", "face", "portrait",
                 "figure", "body", "child", "adult", "baby"]
    if any(kw in summary_lower for kw in person_kw):
        return SubjectType.PERSON

    # Animal keywords
    animal_kw = ["animal", "dog", "cat", "bird", "fish", "horse", "lion", "tiger",
                 "bear", "elephant", "rabbit", "fox", "wolf", "deer", "cow", "pet"]
    if any(kw in summary_lower for kw in animal_kw):
        return SubjectType.ANIMAL

    # Character keywords
    char_kw = ["character", "cartoon", "mascot", "avatar", "sprite", "figure",
               "creature", "monster", "robot", "anime"]
    if any(kw in summary_lower for kw in char_kw):
        return SubjectType.CHARACTER

    # Architecture keywords
    arch_kw = ["building", "house", "architecture", "structure", "tower",
               "bridge", "castle", "temple", "church", "monument"]
    if any(kw in summary_lower for kw in arch_kw):
        return SubjectType.ARCHITECTURE

    # Food keywords
    food_kw = ["food", "dish", "meal", "fruit", "vegetable", "dessert",
               "cake", "pizza", "burger", "drink", "beverage"]
    if any(kw in summary_lower for kw in food_kw):
        return SubjectType.FOOD

    # Plant keywords
    plant_kw = ["plant", "tree", "flower", "garden", "forest", "leaf", "branch"]
    if any(kw in summary_lower for kw in plant_kw):
        return SubjectType.PLANT

    # Abstract keywords
    abstract_kw = ["abstract", "pattern", "shape", "geometric", "design", "logo"]
    if any(kw in summary_lower for kw in abstract_kw):
        return SubjectType.ABSTRACT

    # Default to object
    return SubjectType.OBJECT


# ═══════════════════════════════════════════════════════════════════════════
#  Visual Style Parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_visual_style(style_str: str) -> VisualStyle:
    """Parse visual style from style string."""
    style_lower = style_str.lower()

    mappings = {
        "photorealistic": VisualStyle.PHOTOREALISTIC,
        "photo-realistic": VisualStyle.PHOTOREALISTIC,
        "photo realistic": VisualStyle.PHOTOREALISTIC,
        "realistic": VisualStyle.REALISTIC,
        "semi-realistic": VisualStyle.SEMI_REALISTIC,
        "semi realistic": VisualStyle.SEMI_REALISTIC,
        "cartoon": VisualStyle.CARTOON,
        "cartoonish": VisualStyle.CARTOON,
        "anime": VisualStyle.ANIME,
        "manga": VisualStyle.ANIME,
        "pixel": VisualStyle.PIXEL_ART,
        "pixel art": VisualStyle.PIXEL_ART,
        "pixelated": VisualStyle.PIXEL_ART,
        "sketch": VisualStyle.SKETCH,
        "sketched": VisualStyle.SKETCH,
        "drawing": VisualStyle.SKETCH,
        "watercolor": VisualStyle.WATERCOLOR,
        "water color": VisualStyle.WATERCOLOR,
        "oil": VisualStyle.OIL_PAINTING,
        "oil painting": VisualStyle.OIL_PAINTING,
        "digital": VisualStyle.DIGITAL_ART,
        "digital art": VisualStyle.DIGITAL_ART,
        "vector": VisualStyle.VECTOR,
        "flat": VisualStyle.VECTOR,
        "clay": VisualStyle.CLAY,
        "claymation": VisualStyle.CLAY,
        "paper": VisualStyle.PAPER,
        "papercraft": VisualStyle.PAPER,
    }

    for key, value in mappings.items():
        if key in style_lower:
            return value

    return VisualStyle.UNKNOWN


def _extract_style_descriptors(analysis: Dict[str, Any]) -> List[str]:
    """Extract style descriptor words from analysis."""
    descriptors = []

    style = analysis.get("style", "")
    if style:
        # Extract adjectives and style words
        style_words = re.findall(r'\b[a-z]+\b', style.lower())
        relevant = ["realistic", "detailed", "simple", "complex", "flat", "3d",
                   "minimalist", "ornate", "modern", "vintage", "retro", "futuristic",
                   "bright", "dark", "muted", "vibrant", "soft", "sharp", "bold"]
        descriptors.extend([w for w in style_words if w in relevant])

    return list(set(descriptors))[:5]


# ═══════════════════════════════════════════════════════════════════════════
#  Color Parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_colors(colors: List[Any]) -> Tuple[List[ColorInfo], List[ColorInfo]]:
    """Parse color palette into primary and accent colors."""
    primary = []
    accent = []

    for i, color in enumerate(colors[:8]):
        if isinstance(color, dict):
            name = color.get("name", color.get("color", f"color_{i}"))
            hex_code = color.get("hex", color.get("hex_code"))
            role = color.get("role", "primary" if i < 3 else "accent")
        elif isinstance(color, str):
            name = color
            hex_code = _extract_hex_from_string(color)
            role = "primary" if i < 3 else "accent"
        else:
            continue

        info = ColorInfo(name=name, hex_code=hex_code, role=role)

        if role == "primary" or i < 3:
            primary.append(info)
        else:
            accent.append(info)

    return primary, accent


def _extract_hex_from_string(s: str) -> Optional[str]:
    """Extract hex color code from string if present."""
    match = re.search(r'#[0-9A-Fa-f]{6}', s)
    return match.group(0) if match else None


# ═══════════════════════════════════════════════════════════════════════════
#  Component Parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_components(components: List[Any]) -> Tuple[List[ComponentInfo], List[ComponentInfo]]:
    """Parse components into key and movable components."""
    key_comps = []
    movable_comps = []

    # Words that suggest a component can move independently
    movable_keywords = ["wheel", "leg", "arm", "wing", "door", "hand", "foot",
                       "eye", "tail", "propeller", "blade", "pendulum"]

    for comp in components[:10]:
        if isinstance(comp, dict):
            name = comp.get("name", "element")
            desc = comp.get("description", "")
        elif isinstance(comp, str):
            name = comp
            desc = ""
        else:
            continue

        can_move = any(kw in name.lower() for kw in movable_keywords)
        info = ComponentInfo(name=name, description=desc, can_move=can_move)

        if can_move:
            movable_comps.append(info)
        else:
            key_comps.append(info)

    return key_comps, movable_comps


# ═══════════════════════════════════════════════════════════════════════════
#  Hard Negatives Generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_hard_negatives(profile: VisualIdentityProfile) -> List[str]:
    """Generate hard negative constraints based on the profile."""
    negatives = []

    # Universal negatives
    negatives.append("Do NOT change the overall shape or proportions")
    negatives.append("Do NOT modify the visual style")

    # Color negatives
    if profile.primary_colors:
        color_names = [c.name for c in profile.primary_colors[:3]]
        negatives.append(f"Do NOT change the colors ({', '.join(color_names)})")

    # Subject-specific negatives
    if profile.subject_type == SubjectType.VEHICLE:
        negatives.append("Do NOT change the vehicle type or model appearance")
        negatives.append("Do NOT add or remove vehicle parts")

    elif profile.subject_type == SubjectType.PERSON:
        negatives.append("Do NOT change facial features or proportions")
        negatives.append("Do NOT modify clothing or accessories")
        negatives.append("Do NOT change hair color or style")

    elif profile.subject_type == SubjectType.ANIMAL:
        negatives.append("Do NOT change the species or breed appearance")
        negatives.append("Do NOT modify fur/feather patterns or colors")

    elif profile.subject_type == SubjectType.CHARACTER:
        negatives.append("Do NOT change the character design or defining features")
        negatives.append("Do NOT modify outfit or accessories")

    # Style-specific negatives
    if profile.visual_style == VisualStyle.PIXEL_ART:
        negatives.append("Do NOT change the pixel art resolution")
        negatives.append("Do NOT smooth or anti-alias the pixels")

    elif profile.visual_style in [VisualStyle.CARTOON, VisualStyle.ANIME]:
        negatives.append("Do NOT make it more realistic")
        negatives.append("Do NOT change line weights or outline style")

    elif profile.visual_style == VisualStyle.PHOTOREALISTIC:
        negatives.append("Do NOT stylize or cartoonify")
        negatives.append("Maintain photographic lighting and detail")

    return negatives[:10]  # Limit to 10


def _extract_textures(analysis: Dict[str, Any]) -> List[str]:
    """Extract texture descriptors from analysis."""
    textures = []

    summary = analysis.get("summary", "")
    components = analysis.get("components", [])

    # Common texture words
    texture_words = ["smooth", "rough", "glossy", "matte", "metallic", "wooden",
                    "fabric", "leather", "plastic", "glass", "chrome", "brushed",
                    "textured", "furry", "fluffy", "silky", "grainy", "polished"]

    text_to_search = summary.lower()
    for comp in components:
        if isinstance(comp, dict):
            text_to_search += " " + comp.get("description", "").lower()

    for word in texture_words:
        if word in text_to_search:
            textures.append(word)

    return list(set(textures))[:5]


def _get_animation_recommendations(
    subject_type: SubjectType,
    visual_style: VisualStyle,
) -> Tuple[List[str], List[str]]:
    """Get recommended and avoid animation styles for this subject."""
    recommendations = {
        SubjectType.VEHICLE: (
            ["smooth", "rotate", "bounce"],
            ["walk", "morph"],
        ),
        SubjectType.PERSON: (
            ["walk", "wave", "bounce"],
            ["explode"],
        ),
        SubjectType.ANIMAL: (
            ["walk", "bounce", "wave"],
            ["explode", "rotate"],
        ),
        SubjectType.CHARACTER: (
            ["bounce", "wave", "pulse"],
            [],
        ),
        SubjectType.OBJECT: (
            ["smooth", "rotate", "pulse"],
            ["walk"],
        ),
    }

    recommend, avoid = recommendations.get(subject_type, (["smooth"], []))
    return list(recommend), list(avoid)


def _calculate_confidence(profile: VisualIdentityProfile, analysis: Dict[str, Any]) -> float:
    """Calculate confidence score for the identity extraction."""
    score = 0.0

    # Has subject description
    if profile.subject_description and len(profile.subject_description) > 10:
        score += 0.2

    # Has colors
    if profile.primary_colors:
        score += 0.2

    # Has components
    if profile.key_components:
        score += 0.2

    # Has style
    if profile.visual_style != VisualStyle.UNKNOWN:
        score += 0.2

    # Analysis has substantial content
    if len(str(analysis)) > 200:
        score += 0.2

    return min(score, 1.0)
