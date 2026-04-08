"""Pure-Python SVG rendering for compute selection display.

These renderers produce static SVG strings/files that visualize which
species, reactions, and pressure-dependent networks T3 selected to be
computed. The output is deterministic and persisted as artifacts under
``models/`` so it can be re-served without recomputation.
"""

from html import escape
from pathlib import Path

from carmel.schemas.diagnostics import (
    PDepNetworkSelection,
    ReactionSelection,
    SpeciesSelection,
)
from carmel.services.artifacts import write_text


def _svg_header(width: int, height: int) -> str:
    """Return the opening tag of an SVG document."""
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def render_species_svg(species: list[SpeciesSelection]) -> str:
    """Render a row of species as labeled rounded rectangles.

    Args:
        species: The species to render.

    Returns:
        An SVG document as a string.
    """
    if not species:
        return _svg_header(200, 60) + (
            '<text x="100" y="35" text-anchor="middle" fill="#666" font-family="sans-serif" '
            'font-size="14">no species selected</text></svg>'
        )
    box_w = 140
    box_h = 60
    gap = 16
    cols = max(1, min(4, len(species)))
    rows = (len(species) + cols - 1) // cols
    width = cols * box_w + (cols + 1) * gap
    height = rows * box_h + (rows + 1) * gap + 20
    parts = [_svg_header(width, height)]
    parts.append(
        '<text x="10" y="18" font-family="sans-serif" font-size="14" font-weight="bold" '
        'fill="#333">Species selected to be computed</text>'
    )
    for idx, sp in enumerate(species):
        col = idx % cols
        row = idx // cols
        x = gap + col * (box_w + gap)
        y = 30 + gap + row * (box_h + gap)
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" ry="8" '
            f'fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x + box_w / 2}" y="{y + box_h / 2 - 4}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" fill="#0d47a1" font-weight="bold">'
            f"{escape(sp.label)}</text>"
        )
        if sp.smiles:
            parts.append(
                f'<text x="{x + box_w / 2}" y="{y + box_h / 2 + 14}" text-anchor="middle" '
                f'font-family="monospace" font-size="11" fill="#37474f">'
                f"{escape(sp.smiles[:18])}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def render_reactions_svg(reactions: list[ReactionSelection]) -> str:
    """Render a list of reactions as reactant→product arrows.

    Args:
        reactions: The reactions to render.

    Returns:
        An SVG document as a string.
    """
    if not reactions:
        return _svg_header(300, 60) + (
            '<text x="150" y="35" text-anchor="middle" fill="#666" font-family="sans-serif" '
            'font-size="14">no reactions selected</text></svg>'
        )
    width = 720
    row_h = 36
    height = 30 + len(reactions) * row_h + 20
    parts = [_svg_header(width, height)]
    parts.append(
        '<text x="10" y="18" font-family="sans-serif" font-size="14" font-weight="bold" '
        'fill="#333">Reactions selected to be computed</text>'
    )
    for idx, rxn in enumerate(reactions):
        y = 30 + idx * row_h + row_h // 2
        reactant_str = " + ".join(rxn.reactants) or "?"
        product_str = " + ".join(rxn.products) or "?"
        parts.append(
            f'<text x="20" y="{y + 5}" font-family="monospace" font-size="13" fill="#1b5e20">'
            f"{escape(reactant_str)}</text>"
        )
        parts.append(
            f'<text x="320" y="{y + 5}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="16" fill="#444">→</text>'
        )
        parts.append(
            f'<text x="360" y="{y + 5}" font-family="monospace" font-size="13" fill="#bf360c">'
            f"{escape(product_str)}</text>"
        )
        if rxn.reason:
            parts.append(
                f'<text x="{width - 10}" y="{y + 5}" text-anchor="end" font-family="sans-serif" '
                f'font-size="11" fill="#888">{escape(rxn.reason[:60])}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def render_pdep_networks_svg(networks: list[PDepNetworkSelection]) -> str:
    """Render PDep networks as small radial node graphs.

    Args:
        networks: The PDep networks to render.

    Returns:
        An SVG document as a string.
    """
    if not networks:
        return _svg_header(300, 60) + (
            '<text x="150" y="35" text-anchor="middle" fill="#666" font-family="sans-serif" '
            'font-size="14">no PDep networks selected</text></svg>'
        )
    panel_w = 220
    panel_h = 180
    cols = min(3, len(networks))
    rows = (len(networks) + cols - 1) // cols
    width = cols * panel_w + 20
    height = rows * panel_h + 40
    parts = [_svg_header(width, height)]
    parts.append(
        '<text x="10" y="18" font-family="sans-serif" font-size="14" font-weight="bold" '
        'fill="#333">Pressure-dependent networks selected to be computed</text>'
    )
    for idx, net in enumerate(networks):
        col = idx % cols
        row = idx // cols
        cx = 10 + col * panel_w + panel_w / 2
        cy = 30 + row * panel_h + panel_h / 2
        parts.append(
            f'<rect x="{10 + col * panel_w}" y="{30 + row * panel_h}" '
            f'width="{panel_w - 10}" height="{panel_h - 10}" rx="6" ry="6" '
            f'fill="#fff8e1" stroke="#f57f17" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{cx}" y="{30 + row * panel_h + 18}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#e65100" font-weight="bold">'
            f"{escape(net.network_id)}</text>"
        )
        n_species = max(1, len(net.species))
        radius = 50
        import math

        for i, sp in enumerate(net.species):
            angle = 2 * math.pi * i / n_species - math.pi / 2
            sx = cx + radius * math.cos(angle)
            sy = cy + radius * math.sin(angle)
            parts.append(f'<line x1="{cx}" y1="{cy}" x2="{sx}" y2="{sy}" stroke="#bf360c" stroke-width="1"/>')
            parts.append(f'<circle cx="{sx}" cy="{sy}" r="14" fill="#ffe0b2" stroke="#e65100" stroke-width="1.5"/>')
            parts.append(
                f'<text x="{sx}" y="{sy + 4}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="10" fill="#bf360c">{escape(sp[:6])}</text>'
            )
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="#e65100"/>')
    parts.append("</svg>")
    return "".join(parts)


def write_selection_svgs(
    models_dir: Path,
    species: list[SpeciesSelection],
    reactions: list[ReactionSelection],
    networks: list[PDepNetworkSelection],
) -> dict[str, Path]:
    """Render and persist all three selection SVGs.

    Args:
        models_dir: Target directory (typically ``workspace/models/``).
        species: Species to render.
        reactions: Reactions to render.
        networks: PDep networks to render.

    Returns:
        Mapping of artifact name to file path.
    """
    paths = {
        "species": models_dir / "species_selection.svg",
        "reactions": models_dir / "reactions_selection.svg",
        "pdep_networks": models_dir / "pdep_networks_selection.svg",
    }
    write_text(paths["species"], render_species_svg(species))
    write_text(paths["reactions"], render_reactions_svg(reactions))
    write_text(paths["pdep_networks"], render_pdep_networks_svg(networks))
    return paths
