"""HTML builder for the real-time N-body gravitational simulation.

This module has zero UI imports. It produces a self-contained HTML string
that embeds a vanilla-JS physics engine and canvas renderer. Marimo renders
it via mo.Html().

mo.Html() injects content via innerHTML, which browsers refuse to execute
scripts from. The fix: build a complete HTML document and embed it as an
<iframe srcdoc="..."> — iframes get a real document context where <script>
tags execute normally.

The n_body.js engine handles:
  - N-body gravitational dynamics (Runge-Kutta 4th-order integration)
  - Softening parameter for numerical stability
  - Canvas rendering with trails and labels
  - Real-time live animation at 60 FPS
"""

import html as _html
import importlib.resources
import json

from simulations.physics.gravity.params import GravityParams


def _load_nbody_js() -> str:
    pkg = importlib.resources.files("simulations.physics.gravity")
    return (pkg / "assets" / "n_body.js").read_text(encoding="utf-8")


def build_gravity_html(params: GravityParams) -> str:
    """Return an <iframe srcdoc="..."> that runs a live N-body gravitational simulation.

    The srcdoc is a complete HTML document containing:
    - A <canvas> sized to params.canvas_width × params.canvas_height
    - An inline <script> containing the vanilla-JS physics engine and renderer
    - A second <script> that initialises the simulation with params as a JSON config

    Multiple bodies can be simulated with their own masses, positions, velocities,
    and colors. Each body renders as a circle with (optionally) a trailing arc.
    The simulation uses softening for numerical stability and supports both
    sun-centered and centroid-centered views.

    Raises:
        ValueError: if params.validate() fails.
    """
    params.validate()
    config_json = json.dumps(params.to_dict())
    js = _load_nbody_js()

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0d1a; overflow: hidden; }}
  canvas {{ display: block; }}
</style>
</head>
<body>
<canvas id="c" width="{params.canvas_width}" height="{params.canvas_height}"></canvas>
<script>
{js}
</script>
<script>
window.NBody.start({config_json});
</script>
</body>
</html>"""

    # Escape for the srcdoc attribute value (replaces ", &, <, >)
    srcdoc = _html.escape(inner_html, quote=True)

    w = params.canvas_width
    h = params.canvas_height
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'width="{w}" height="{h}" '
        f'frameborder="0" scrolling="no" '
        f'style="border:none;display:block;border-radius:4px;"></iframe>'
    )
