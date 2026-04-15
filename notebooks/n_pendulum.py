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
    mo.Html("""<style>
.tip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  position: relative;
  cursor: help;
  color: #6b7280;
  font-size: 0.7em;
  border: 1px solid #9ca3af;
  border-radius: 50%;
  width: 1.5em;
  height: 1.5em;
  margin-left: 4px;
  vertical-align: middle;
  font-style: normal;
  user-select: none;
}
.tip::after {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: #1f2937;
  color: #f9fafb;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 1.2rem;
  width: 240px;
  white-space: normal;
  line-height: 1.4;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s ease;
  z-index: 9999;
  text-align: left;
}
.tip:hover::after { opacity: 1; }
</style>""")
    return


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
    n_links_slider = mo.ui.slider(start=1, stop=5, step=1, value=2, label='Links <span class="tip" data-tip="Number of rigid links in the pendulum chain. 1 = simple pendulum (periodic); 2+ = chaotic for large angles.">?</span>')
    g_slider = mo.ui.slider(start=0, stop=20, step=0.1, value=9.81, label='Gravity (m/s²) <span class="tip" data-tip="Gravitational acceleration. 0 = zero-g (pendulum drifts freely); 9.81 = Earth; 20 = strong pull.">?</span>')
    scale_slider = mo.ui.slider(start=50, stop=300, step=10, value=180, label='Scale (px/m) <span class="tip" data-tip="Pixels per metre. Controls how large the pendulum appears on screen. Does not affect the physics.">?</span>')
    trail_toggle = mo.ui.checkbox(value=True, label='Show trails <span class="tip" data-tip="Draw the path traced by the tip of the last link. Useful for visualising chaotic vs periodic motion.">?</span>')
    trail_length_slider = mo.ui.slider(start=50, stop=500, step=50, value=200, label='Trail length <span class="tip" data-tip="Number of recent tip positions to keep in the trail. Longer trails show more history but can obscure the current position.">?</span>')
    ic_textarea = mo.ui.text_area(
        value="120, -60\n120.2, -60\n119.8, -60.1\n",
        label='Initial angles (°) — one pendulum per line, N angles per line <span class="tip" data-tip="Each line defines one pendulum. Provide N comma-separated angles (degrees from downward vertical) for an N-link chain. Multiple nearly-identical lines visualise chaos: tiny differences diverge rapidly.">?</span>',
    )

    mo.vstack([
        mo.hstack([n_links_slider, g_slider, scale_slider], justify="center"),
        mo.hstack([trail_toggle, trail_length_slider], justify="center"),
        mo.hstack([ic_textarea], justify="center"),
    ])
    return (
        g_slider,
        ic_textarea,
        n_links_slider,
        scale_slider,
        trail_length_slider,
        trail_toggle,
    )


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
    _ics, _omegas = parse_ic_text(ic_textarea.value, _n, 10)

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
