"""Convert ``docs/_static/images/mpstab_pipeline.pdf`` into the light and dark
SVG figures shown on the documentation landing page.

The PDF is turned into an SVG with text drawn as paths. The math labels are
black PNGs shaped by a PNG mask: the black image is replaced by a rectangle so
that its colour can be themed like the rest of the drawing. The light variant
keeps the original colours; the dark variant swaps black for a light grey and
the pastel fills for darker tones. Furo shows one or the other according to the
page theme (``only-light`` / ``only-dark`` image classes).

Requires PyMuPDF (``pip install pymupdf``). Run from the repository root::

    python docs/scripts/pipeline_figure_to_svg.py
"""

import re
from pathlib import Path

import fitz

IMAGES = Path(__file__).resolve().parents[1] / "_static" / "images"
SOURCE = IMAGES / "mpstab_pipeline.pdf"

DARK_INK = "#a9acb2"
DARK_PALETTE = {
    "#000000": DARK_INK,
    "#211715": DARK_INK,
    # panel backgrounds
    "#fffef2": "#1d1d18",
    "#f2f5ff": "#191c27",
    "#fff3f2": "#221b1d",
    "#fff3f3": "#271d20",
    "#e3e9ff": "#212739",
    "#eaeaea": "#26272b",
    "#ffffff": "#1e1f22",
    # gates and states
    "#fdffae": "#5e5c2c",
    "#ffaeb0": "#6c383c",
    "#aec0ff": "#3a4a80",
    "#cbffd0": "#2c5834",
    "#ebccff": "#523a6c",
}

IMAGE = re.compile(
    r'<image id="\w+" width="(\d+)" height="(\d+)" '
    r'xlink:href="data:image/png;base64,[^"]*"\s*/>'
)


def replace_masked_images(svg):
    """Replace every embedded PNG outside a ``<mask>`` with a black rectangle.

    Args:
        svg (str): SVG source.

    Returns:
        str: SVG source in which only the mask PNGs are left.
    """
    pieces, position = [], 0
    for match in IMAGE.finditer(svg):
        in_mask = svg.rfind("<mask", 0, match.start()) > svg.rfind(
            "</mask>", 0, match.start()
        )
        pieces.append(svg[position : match.start()])
        pieces.append(
            match[0]
            if in_mask
            else f'<rect width="{match[1]}" height="{match[2]}" fill="#000000"/>'
        )
        position = match.end()
    return "".join(pieces) + svg[position:]


def themed(svg, palette, ink):
    """Recolour an SVG.

    Args:
        svg (str): SVG source.
        palette (dict): Map from original to new ``#rrggbb`` colours.
        ink (str): Fill inherited by elements that do not set one.

    Returns:
        str: The recoloured SVG source.
    """
    svg = re.sub(r'"(#[0-9a-f]{6})"', lambda m: f'"{palette.get(m[1], m[1])}"', svg)
    return svg.replace("<svg ", f'<svg fill="{ink}" ', 1)


def main():
    """Write the light and dark SVG figures next to the PDF source."""
    svg = fitz.open(SOURCE)[0].get_svg_image(text_as_path=True)
    svg = re.sub(r'\s*xmlns:inkscape="[^"]*"', "", svg)
    svg = replace_masked_images(svg)
    (IMAGES / "mpstab_pipeline_light.svg").write_text(themed(svg, {}, "#000000"))
    (IMAGES / "mpstab_pipeline_dark.svg").write_text(
        themed(svg, DARK_PALETTE, DARK_INK)
    )


if __name__ == "__main__":
    main()
