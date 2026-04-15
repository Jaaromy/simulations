"""HTML builder for the real-time N-link pendulum simulation.

This module has zero UI imports. It produces a self-contained HTML string
that embeds a vanilla-JS physics engine and canvas renderer. Marimo renders
it via mo.Html().

mo.Html() injects content via innerHTML, which browsers refuse to execute
scripts from. The fix: build a complete HTML document and embed it as an
<iframe srcdoc="..."> — iframes get a real document context where <script>
tags execute normally.

The n_pendulum.js engine handles:
  - Lagrangian N-link pendulum dynamics (Runge-Kutta 4th-order integration)
  - Generalized mass matrix with centripetal and gravity terms
  - Canvas rendering with trails and bobs
  - Real-time live animation at 60 FPS
"""

import html as _html
import importlib.resources
import json

from simulations.physics.n_pendulum.params import NPendulumParams


def _load_npendulum_js() -> str:
    pkg = importlib.resources.files("simulations.physics.n_pendulum")
    return (pkg / "assets" / "n_pendulum.js").read_text(encoding="utf-8")


def build_n_pendulum_html(params: NPendulumParams) -> str:
    """Return an <iframe srcdoc="..."> that runs a live N-link pendulum simulation.

    The srcdoc is a complete HTML document containing:
    - A <canvas> sized to params.canvas_width × params.canvas_height
    - An inline <script> containing the vanilla-JS physics engine and renderer
    - A second <script> that initialises the simulation with params as a JSON config

    Multiple pendulums can be simulated in parallel, each with independent
    initial conditions but sharing physics parameters (masses, lengths, g).
    Each pendulum renders as an arm with bobs and (optionally) a trailing arc.

    Raises:
        ValueError: if params.validate() fails.
    """
    params.validate()
    config_json = json.dumps(params.to_dict())
    js = _load_npendulum_js()

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
window.NPendulum.start({config_json});
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
