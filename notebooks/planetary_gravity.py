import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
async def _():
    wasm_ready = True
    try:
        import micropip
        await micropip.install("../../wheels/simulations-0.1.0-py3-none-any.whl")
    except Exception:
        pass
    return (wasm_ready,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Planetary Gravity Simulation
    Real-time N-body gravity simulation using Velocity-Verlet integration.
    Default scenario: our solar system (Sun + 8 planets + Moon). Adjust controls to explore. Click and drag on the canvas to add new bodies.
    """)
    return


@app.cell
def _(mo):
    time_warp_slider = mo.ui.slider(start=1000, stop=10000000, step=1000, value=86400, label='Time warp (s/s) <abbr title="Simulation speed: sim-seconds per real-second. Default 86400 = 1 day/s (Earth orbits in ~6 min).">[?]</abbr>')
    view_scale_slider = mo.ui.slider(start=0.5, stop=35, step=0.5, value=3.2, label='View scale (AU) — half-width of viewport in AU.')
    dt_slider = mo.ui.slider(start=60, stop=86400, step=60, value=3600, label='Physics timestep (s) — smaller = more accurate but slower.')
    trail_toggle = mo.ui.checkbox(value=True, label='Show trails')
    labels_toggle = mo.ui.checkbox(value=True, label='Show labels')
    log_radius_toggle = mo.ui.checkbox(value=True, label='Log radius scale — compresses visual sizes so planets are visible beside the Sun.')
    merge_toggle = mo.ui.checkbox(value=True, label='Inelastic merges — bodies combine on contact.')
    view_center_dropdown = mo.ui.dropdown(options=["sun", "centroid"], value="sun", label='Camera center')
    preset_dropdown = mo.ui.dropdown(options=["solar_system", "inner_only", "binary", "figure8_3body"], value="solar_system", label='Preset scenario')
    return (
        dt_slider,
        labels_toggle,
        log_radius_toggle,
        merge_toggle,
        preset_dropdown,
        time_warp_slider,
        trail_toggle,
        view_center_dropdown,
        view_scale_slider,
    )


@app.cell
def _(mo, preset_dropdown):
    from simulations.physics.gravity.params import solar_system_preset, bodies_to_text, BodyIC
    import math

    def _inner_only():
        _bodies = solar_system_preset()
        return [b for b in _bodies if b.name in ("Sun", "Mercury", "Venus", "Earth", "Mars", "Luna")]

    def _binary():
        _G = 6.67430e-11
        _M = 1.989e30
        _a = 1.0e11
        _v = math.sqrt(_G * _M / (2 * _a))
        return [
            BodyIC("Star A", _M, -_a, 0.0, 0.0, -_v, 6.957e8, "#FDB813"),
            BodyIC("Star B", _M,  _a, 0.0, 0.0,  _v, 6.957e8, "#4B9CD3"),
        ]

    def _figure8():
        _G = 6.67430e-11
        _M = 1.989e30
        _scale = 1.5e11
        _vscale = math.sqrt(_G * _M / _scale)  # ~29744 m/s — self-consistent with G, M, scale
        return [
            BodyIC("Body A", _M, -0.97000436*_scale,  0.24308753*_scale,  0.93240737/2*_vscale,  0.86473146/2*_vscale, 6.957e8, "#FF6B6B"),
            BodyIC("Body B", _M,  0.97000436*_scale, -0.24308753*_scale,  0.93240737/2*_vscale,  0.86473146/2*_vscale, 6.957e8, "#4ECDC4"),
            BodyIC("Body C", _M,  0.0,                0.0,               -0.93240737*_vscale,   -0.86473146*_vscale,   6.957e8, "#45B7D1"),
        ]

    _presets = {
        "solar_system": solar_system_preset,
        "inner_only": _inner_only,
        "binary": _binary,
        "figure8_3body": _figure8,
    }
    _bodies = _presets[preset_dropdown.value]()
    _default_text = bodies_to_text(_bodies)

    ic_textarea = mo.ui.text_area(
        value=_default_text,
        label='Bodies — one per line: name, mass_kg, x_AU, y_AU, vx_km_s, vy_km_s, radius_km, color <abbr title="Edit to add custom bodies. Lines starting with # are comments.">[?]</abbr>',
    )
    return (ic_textarea,)


@app.cell
def _(
    dt_slider,
    ic_textarea,
    labels_toggle,
    log_radius_toggle,
    merge_toggle,
    mo,
    preset_dropdown,
    time_warp_slider,
    trail_toggle,
    view_center_dropdown,
    view_scale_slider,
):
    mo.vstack([
        mo.hstack([time_warp_slider, view_scale_slider, dt_slider], justify="center"),
        mo.hstack([trail_toggle, labels_toggle, log_radius_toggle, merge_toggle], justify="center"),
        mo.hstack([view_center_dropdown, preset_dropdown], justify="center"),
        mo.hstack([ic_textarea], justify="center"),
    ])
    return


@app.cell
def _(
    dt_slider,
    ic_textarea,
    labels_toggle,
    log_radius_toggle,
    merge_toggle,
    mo,
    time_warp_slider,
    trail_toggle,
    view_center_dropdown,
    view_scale_slider,
    wasm_ready,
):
    from simulations.physics.gravity.params import GravityParams, parse_ic_text
    from simulations.physics.gravity.widget import build_gravity_html

    _bodies = parse_ic_text(ic_textarea.value)
    _params = GravityParams(
        bodies=_bodies,
        time_warp=float(time_warp_slider.value),
        view_scale_au=float(view_scale_slider.value),
        dt_s=float(dt_slider.value),
        trail_enabled=trail_toggle.value,
        show_labels=labels_toggle.value,
        log_radius_scale=log_radius_toggle.value,
        merge_enabled=merge_toggle.value,
        view_center=view_center_dropdown.value,
    )
    mo.hstack([mo.Html(build_gravity_html(_params))], justify="center")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Physics

    This simulation integrates Newton's law of gravitation for N bodies using the **Velocity-Verlet** (symplectic) integrator.

    For each pair of bodies i, j, the gravitational acceleration on body i is:

    > **aᵢ = G Σⱼ mⱼ (rⱼ − rᵢ) / (|rᵢⱼ|² + ε²)^(3/2)**

    where ε is a softening length to prevent singularities at close range.

    **Velocity-Verlet** is a symplectic integrator: it conserves a modified energy exactly, leading to bounded energy drift (≤ 0.01% over many orbits) versus Euler's unbounded drift. This makes it far more accurate for orbital simulations.

    **Kepler's Third Law** (sanity check): for a circular orbit of radius a around mass M, the period is T = 2π√(a³/GM). Earth's period should be ≈365.25 days — verify by watching the simulation.

    **Inelastic merges**: when two bodies' surfaces touch (|rᵢⱼ| < rᵢ + rⱼ), they merge conserving mass and momentum. Volume (and hence radius) is also conserved: r_new = (rᵢ³ + rⱼ³)^(1/3).
    """)
    return


if __name__ == "__main__":
    app.run()
