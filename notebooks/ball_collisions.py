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
    # 2D Ball Collisions

    Real-time rigid-body physics via **Matter.js**.  Adjust any slider to restart
    the simulation with the new parameters.
    """)
    return


@app.cell
def _(mo):
    n_balls_slider = mo.ui.slider(
        start=1, stop=80, step=1, value=15,
        label='Balls <span class="tip" data-tip="Number of balls in the simulation. More balls → more collisions; above ~60 the canvas gets crowded.">?</span>',
    )
    friction_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.05,
        label='Friction <span class="tip" data-tip="Coulomb surface friction coefficient. 0 = frictionless ice; 1 = maximum grip. Applied at contact between balls and walls.">?</span>',
    )
    friction_air_slider = mo.ui.slider(
        start=0.0, stop=0.1, step=0.001, value=0.005,
        label='Air drag <span class="tip" data-tip="Air resistance coefficient applied every tick. 0 = vacuum (no drag); 0.1 = strong drag. Independent of surface friction.">?</span>',
    )
    restitution_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.8,
        label='Restitution <span class="tip" data-tip="Coefficient of restitution (bounciness). 0 = perfectly inelastic (balls stick on impact); 1 = perfectly elastic (no kinetic energy lost).">?</span>',
    )
    speed_slider = mo.ui.slider(
        start=0.0, stop=20.0, step=0.5, value=6.0,
        label='Initial speed <span class="tip" data-tip="Speed of each ball at launch (pixels per engine tick, ~60 Hz). 0 = balls start at rest.">?</span>',
    )
    gravity_slider = mo.ui.slider(
        start=0.0, stop=3.0, step=0.05, value=1.0,
        label='Gravity <span class="tip" data-tip="Vertical gravity scale. 0 = zero-g / billiards mode; 1 = Earth-like; 3 = strong pull.">?</span>',
    )
    mass_max_slider = mo.ui.slider(
        start=1.0, stop=10.0, step=0.5, value=5.0,
        label='Max mass <span class="tip" data-tip="Upper bound of the uniform mass range [1, max]. Ball radius is proportional to mass. Heavier balls are darker purple.">?</span>',
    )
    vectors_toggle = mo.ui.checkbox(
        value=False,
        label='Velocity vectors <span class="tip" data-tip="Draw yellow arrows showing each ball\'s current velocity. Arrow length scales with speed.">?</span>',
    )

    mo.vstack([
        mo.hstack([n_balls_slider, friction_slider, friction_air_slider], justify="center"),
        mo.hstack([restitution_slider, speed_slider, gravity_slider], justify="center"),
        mo.hstack([mass_max_slider, vectors_toggle], justify="center"),
    ])
    return (
        friction_air_slider,
        friction_slider,
        gravity_slider,
        mass_max_slider,
        n_balls_slider,
        restitution_slider,
        speed_slider,
        vectors_toggle,
    )


@app.cell
def _(
    friction_air_slider,
    friction_slider,
    gravity_slider,
    mass_max_slider,
    mo,
    n_balls_slider,
    restitution_slider,
    speed_slider,
    vectors_toggle,
    wasm_ready,
):
    from simulations.physics.collisions.params import CollisionParams
    from simulations.physics.collisions.widget import build_collision_html

    _params = CollisionParams(
        n_balls=n_balls_slider.value,
        friction=friction_slider.value,
        friction_air=friction_air_slider.value,
        restitution=restitution_slider.value,
        mass_min=1.0,
        mass_max=float(mass_max_slider.value),
        speed=float(speed_slider.value),
        gravity_y=float(gravity_slider.value),
        show_velocity_vectors=vectors_toggle.value,
    )

    mo.hstack([mo.Html(build_collision_html(_params))], justify="center")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Physics

    **Coefficient of restitution** `e` governs how much kinetic energy is preserved
    in a collision.  For two balls of mass `m₁`, `m₂` with relative approach speed `u`:

    > post-collision relative speed = e · u

    `e = 1` → perfectly elastic (KE conserved); `e = 0` → perfectly inelastic (balls
    merge).

    **Coulomb friction** applies a tangential impulse at each contact point,
    proportional to the normal force.

    **Air drag** is a separate damping coefficient applied every tick regardless
    of contact.  Setting it to 0 (vacuum) with restitution = 1 gives a perfectly
    conservative system: balls bounce indefinitely with no energy loss.

    **Conservation of momentum** holds in every collision regardless of `e`:

    > m₁v₁ + m₂v₂ = m₁v₁' + m₂v₂'

    Ball radius scales as `r = 5 + mass × 3.5` px so the visual size reflects
    the inertia of each ball.
    """)
    return


if __name__ == "__main__":
    app.run()
