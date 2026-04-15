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
    # N-Link Pendulum
    Real-time N-link pendulum simulation using Lagrangian mechanics and RK4 integration.
    Adjust any control to restart with new parameters.
    """)
    return


@app.cell
def _(mo):
    n_links_slider = mo.ui.slider(start=1, stop=5, step=1, value=2, label='Links <abbr title="Number of rigid links in the pendulum chain. 1 = simple pendulum (periodic); 2+ = chaotic for large angles.">[?]</abbr>')
    g_slider = mo.ui.slider(start=0, stop=20, step=0.1, value=9.81, label='Gravity (m/s²) <abbr title="Gravitational acceleration. 0 = zero-g (pendulum drifts freely); 9.81 = Earth; 20 = strong pull.">[?]</abbr>')
    scale_slider = mo.ui.slider(start=50, stop=300, step=10, value=180, label='Scale (px/m) <abbr title="Pixels per metre. Controls how large the pendulum appears on screen. Does not affect the physics.">[?]</abbr>')
    trail_toggle = mo.ui.checkbox(value=True, label='Show trails <abbr title="Draw the path traced by the tip of the last link. Useful for visualising chaotic vs periodic motion.">[?]</abbr>')
    trail_length_slider = mo.ui.slider(start=50, stop=500, step=50, value=200, label='Trail length <abbr title="Number of recent tip positions to keep in the trail. Longer trails show more history but can obscure the current position.">[?]</abbr>')
    return (g_slider, n_links_slider, scale_slider, trail_length_slider, trail_toggle)


@app.cell
def _(mo, n_links_slider):
    def _default_ic_text(n: int) -> str:
        # Angles alternate sign and halve each link: 120, -60, 30, -15, 7.5
        _base = [round(120.0 * ((-0.5) ** i), 1) for i in range(n)]
        _fmt = lambda angles: ", ".join(str(a) for a in angles)
        _p1 = _base.copy(); _p1[0] = round(_p1[0] + 0.2, 1)
        _p2 = _base.copy(); _p2[0] = round(_p2[0] - 0.2, 1)
        if n > 1:
            _p2[1] = round(_p2[1] - 0.1, 1)
        return f"{_fmt(_base)}\n{_fmt(_p1)}\n{_fmt(_p2)}\n"

    ic_textarea = mo.ui.text_area(
        value=_default_ic_text(n_links_slider.value),
        label='Initial angles (°) — one pendulum per line, N comma-separated angles per line <abbr title="Each line defines one pendulum. Provide N comma-separated angles (degrees from downward vertical) for an N-link chain. Multiple nearly-identical lines visualise chaos: tiny differences diverge rapidly.">[?]</abbr>',
    )
    return (ic_textarea,)


@app.cell
def _(g_slider, ic_textarea, mo, n_links_slider, scale_slider, trail_length_slider, trail_toggle):
    mo.vstack([
        mo.hstack([n_links_slider, g_slider, scale_slider], justify="center"),
        mo.hstack([trail_toggle, trail_length_slider], justify="center"),
        mo.hstack([ic_textarea], justify="center"),
    ])
    return


@app.cell
def _(
    g_slider,
    ic_textarea,
    mo,
    n_links_slider,
    scale_slider,
    trail_length_slider,
    trail_toggle,
    wasm_ready,
):
    from simulations.physics.n_pendulum.params import NPendulumParams, parse_ic_text
    from simulations.physics.n_pendulum.widget import build_n_pendulum_html

    _n = n_links_slider.value
    _n_pendulums = len([l for l in ic_textarea.value.strip().splitlines() if l.strip()]) or 1
    _ics, _omegas = parse_ic_text(ic_textarea.value, _n, _n_pendulums)

    _params = NPendulumParams(
        n_links=_n,
        masses=[1.0] * _n,
        lengths=[1.0 / _n] * _n,
        g=g_slider.value,
        initial_conditions=_ics,
        initial_omegas=_omegas,
        trail_enabled=trail_toggle.value,
        trail_length=trail_length_slider.value,
        scale_px_per_m=float(scale_slider.value),
    )

    mo.hstack([mo.Html(build_n_pendulum_html(_params))], justify="center")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Physics

    This simulation integrates the **N-link pendulum** equations of motion derived via Lagrangian mechanics.

    For N links with masses m₁…mN and lengths l₁…lN, the generalised coordinates are angles θ₁…θN
    measured from the downward vertical. The equations of motion take the form:

    > **M(θ) θ̈ = −C(θ, θ̇) − G(θ)**

    where **M** is the N×N mass matrix, **C** is the Coriolis/centripetal vector, and **G** is the gravity vector.

    **RK4 integration** at dt = 1/240 s provides < 0.1% energy drift over 10 seconds for N ≤ 4.

    **Chaos**: small changes in initial angles (< 1°) lead to completely different trajectories.
    Try multiple pendulums with nearly identical starting angles to visualise this sensitivity.
    """)
    return


if __name__ == "__main__":
    app.run()
