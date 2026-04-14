"""HTML builder for the real-time 2D ball collision simulation.

This module has zero UI imports.  It produces a self-contained HTML string
that embeds a Matter.js physics engine.  Marimo renders it via mo.Html().

mo.Html() injects content via innerHTML, which browsers refuse to execute
scripts from.  The fix: build a complete HTML document and embed it as an
<iframe srcdoc="..."> — iframes get a real document context where <script>
tags execute normally.

Marimo also serves with a Content-Security-Policy that blocks external CDN
scripts in iframes.  Matter.js is therefore bundled as matter.min.js alongside
this module and inlined directly into the srcdoc, making the simulation fully
self-contained with no network requests.

Matter.js handles:
  - Rigid-body circle collisions (SAT narrow phase)
  - Coefficient of restitution (energy loss on impact)
  - Coulomb friction (surface friction + air resistance)
  - Gravity vector
  - Continuous collision detection (no tunnelling at high speeds)
"""

import html as _html
import importlib.resources
import json

from simulations.physics.collisions.params import CollisionParams


def _load_matterjs() -> str:
    pkg = importlib.resources.files("simulations.physics.collisions")
    return (pkg / "matter.min.js").read_text(encoding="utf-8")


def build_collision_html(params: CollisionParams) -> str:
    """Return an <iframe srcdoc="..."> that runs a live Matter.js simulation.

    The srcdoc is a complete HTML document containing:
    - A <canvas> sized to params.canvas_width × params.canvas_height
    - A <script> loading Matter.js 0.20.0 from cdnjs (synchronous, so it
      blocks until Matter is available before the inline script runs)
    - An inline IIFE that initialises the engine with params as a JSON config

    Ball radius is proportional to mass; colour goes sky-blue → purple with
    increasing mass.  Velocity arrows are drawn when show_velocity_vectors=True.

    Raises:
        ValueError: if params.validate() fails.
    """
    params.validate()
    config_json = json.dumps(params.to_dict())
    matterjs = _load_matterjs()
    js = _build_js(config_json)

    toolbar_h = 40  # px; must match #toolbar height in CSS below
    inner_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0d1a; overflow: hidden; }}
  canvas {{ display: block; }}
  #toolbar {{
    height: {toolbar_h}px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #1a1a2e;
    border-top: 1px solid #2a2a4a;
  }}
  #restart {{
    background: rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.75);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px;
    padding: 4px 18px;
    font-size: 13px;
    cursor: pointer;
    letter-spacing: 0.03em;
  }}
  #restart:hover {{
    background: rgba(255,255,255,0.16);
    color: #fff;
  }}
</style>
</head>
<body>
<canvas id="c" width="{params.canvas_width}" height="{params.canvas_height}"></canvas>
<div id="toolbar"><button id="restart">&#8635; Restart</button></div>
<script>{matterjs}</script>
<script>
{js}
</script>
</body>
</html>"""

    # Escape for the srcdoc attribute value (replaces ", &, <, >)
    srcdoc = _html.escape(inner_html, quote=True)

    w = params.canvas_width
    h = params.canvas_height + toolbar_h
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'width="{w}" height="{h}" '
        f'frameborder="0" scrolling="no" '
        f'style="border:none;display:block;border-radius:4px;"></iframe>'
    )


def _build_js(config_json: str) -> str:
    """Return the IIFE that initialises and runs the Matter.js simulation.

    The canvas element has id="c" and already exists in the document when
    this script executes (the CDN <script src> above blocks until loaded).
    """
    return f"""(function () {{
  var config = {config_json};
  var canvasEl = document.getElementById('c');

  // ── Engine ────────────────────────────────────────────────────────────────
  var engine = Matter.Engine.create({{
    enableSleeping:       true,
    positionIterations:   20,
    velocityIterations:   16,
    constraintIterations: 6,
  }});
  engine.gravity.y = config.gravity_y;

  // Sleep only when motion is truly imperceptible — default 0.08 is too coarse
  // and makes bodies snap to a halt. At SLEEP_THRESHOLD the velocity is already
  // visually indistinguishable from zero before sleep freezes it.
  var SLEEP_THRESHOLD = 0.005;
  Matter.Sleeping._motionSleepThreshold = SLEEP_THRESHOLD;

  // ── Renderer ──────────────────────────────────────────────────────────────
  var render = Matter.Render.create({{
    canvas: canvasEl,
    engine: engine,
    options: {{
      width: config.canvas_width,
      height: config.canvas_height,
      wireframes: false,
      background: '#0d0d1a',
      showSleeping: false
    }}
  }});

  // ── Walls ─────────────────────────────────────────────────────────────────
  var WALL_THICKNESS = 60;
  var W = config.canvas_width;
  var H = config.canvas_height;
  var wallOpts = {{ isStatic: true, render: {{ fillStyle: '#2a2a4a' }} }};
  Matter.Composite.add(engine.world, [
    Matter.Bodies.rectangle(W / 2, H + WALL_THICKNESS / 2, W + WALL_THICKNESS * 2, WALL_THICKNESS, wallOpts),
    Matter.Bodies.rectangle(W / 2, -WALL_THICKNESS / 2,    W + WALL_THICKNESS * 2, WALL_THICKNESS, wallOpts),
    Matter.Bodies.rectangle(-WALL_THICKNESS / 2,    H / 2, WALL_THICKNESS, H, wallOpts),
    Matter.Bodies.rectangle(W + WALL_THICKNESS / 2, H / 2, WALL_THICKNESS, H, wallOpts)
  ]);

  // ── Balls ─────────────────────────────────────────────────────────────────
  // Uniform density so visual area ∝ mass — radius = √(mass / (ρπ)).
  // Choosing ρ = 1/(64π) gives r ≈ 8 px for the lightest ball (mass = 1).
  var DENSITY = 1 / (64 * Math.PI);
  // Static friction is always at least STATIC_FRICTION_MULT × kinetic friction
  var STATIC_FRICTION_MULT = 1.5, MIN_STATIC_FRICTION = 0.1;
  // Ball colour gradient: hue 200° (sky-blue) → 280° (purple), lightness drops with mass
  var HUE_START = 200, HUE_RANGE = 80, LIGHTNESS_START = 70, LIGHTNESS_RANGE = 35;

  function spawnBalls() {{
    var result = [];
    for (var i = 0; i < config.n_balls; i++) {{
      var t      = (config.n_balls > 1) ? i / (config.n_balls - 1) : 0.5;
      var mass   = config.mass_min + t * (config.mass_max - config.mass_min);
      var radius = Math.sqrt(mass / (DENSITY * Math.PI));
      var margin = radius + 2;
      var x = margin + Math.random() * (W - 2 * margin);
      var y = margin + Math.random() * (H - 2 * margin);

      var ball = Matter.Bodies.circle(x, y, radius, {{
        restitution:    config.restitution,
        friction:       config.friction,
        frictionStatic: Math.max(config.friction * STATIC_FRICTION_MULT, MIN_STATIC_FRICTION),
        frictionAir:    config.friction_air,
        density:        DENSITY,
      }});

      var hue   = Math.round(HUE_START + t * HUE_RANGE);
      var light = Math.round(LIGHTNESS_START - t * LIGHTNESS_RANGE);
      ball.render.fillStyle   = 'hsl(' + hue + ',65%,' + light + '%)';
      ball.render.strokeStyle = 'rgba(255,255,255,0.15)';
      ball.render.lineWidth   = 1;

      var baseAngle  = Math.random() * 2 * Math.PI;
      var halfSpread = (config.angle_spread / 2) * Math.PI / 180;
      var angle      = baseAngle + (Math.random() - 0.5) * 2 * halfSpread;
      Matter.Body.setVelocity(ball, {{
        x: config.speed * Math.cos(angle),
        y: config.speed * Math.sin(angle)
      }});

      result.push(ball);
    }}
    Matter.Composite.add(engine.world, result);
    return result;
  }}

  var balls = spawnBalls();

  // ── Restart button ────────────────────────────────────────────────────────
  document.getElementById('restart').addEventListener('click', function () {{
    Matter.Composite.clear(engine.world, true); // true = keep static walls
    balls = spawnBalls();
  }});

  // ── Velocity vectors overlay ───────────────────────────────────────────────
  if (config.show_velocity_vectors) {{
    Matter.Events.on(render, 'afterRender', function () {{
      var ctx   = render.context;
      var scale = 8;
      balls.forEach(function (b) {{
        var px = b.position.x, py = b.position.y;
        var vx = b.velocity.x * scale, vy = b.velocity.y * scale;
        if (Math.sqrt(vx * vx + vy * vy) < 0.5) {{ return; }}
        var ex = px + vx, ey = py + vy;
        var ang = Math.atan2(vy, vx), alen = 7;
        ctx.save();
        ctx.strokeStyle = 'rgba(255,240,60,0.85)';
        ctx.fillStyle   = 'rgba(255,240,60,0.85)';
        ctx.lineWidth   = 1.5;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - alen * Math.cos(ang - Math.PI / 6),
                   ey - alen * Math.sin(ang - Math.PI / 6));
        ctx.lineTo(ex - alen * Math.cos(ang + Math.PI / 6),
                   ey - alen * Math.sin(ang + Math.PI / 6));
        ctx.closePath(); ctx.fill();
        ctx.restore();
      }});
    }});
  }}

  // ── Run ───────────────────────────────────────────────────────────────────
  Matter.Render.run(render);
  // 120 Hz physics: half the per-step displacement → shallower penetrations
  // → solver converges before residual velocity can propagate as a wave.
  var runner = Matter.Runner.create({{ delta: 1000 / 120 }});
  Matter.Runner.run(runner, engine);
}})();"""
