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
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np


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
    # Double Pendulum
    """)
    return


@app.cell
def _(mo):
    theta1_slider = mo.ui.slider(
        start=-180, stop=180, step=1, value=120,
        label='θ₁ — upper bob angle (°) <span class="tip" data-tip="Angle of the upper rod from the downward vertical. 0° = hanging straight down. 180° = pointing straight up.">?</span>',
    )
    theta2_slider = mo.ui.slider(
        start=-180, stop=180, step=1, value=-20,
        label='θ₂ — lower bob angle (°) <span class="tip" data-tip="Angle of the lower rod from the downward vertical. Setting this close to θ₁ gives regular motion; large differences produce chaos.">?</span>',
    )
    t_end_slider = mo.ui.slider(
        start=5, stop=60, step=5, value=30,
        label='Duration (s) <span class="tip" data-tip="How long to simulate. Longer runs reveal more of the chaotic trajectory.">?</span>',
    )
    chaos_toggle = mo.ui.checkbox(
        value=True,
        label='Show chaos comparison <span class="tip" data-tip="Runs a second pendulum with θ₁ perturbed by 0.01°. Both trajectories start indistinguishable but diverge exponentially — visualising sensitive dependence on initial conditions.">?</span>',
    )
    run_button = mo.ui.run_button(label="Run simulation")

    mo.vstack([
        mo.hstack([theta1_slider, theta2_slider, t_end_slider], justify="center"),
        mo.hstack([chaos_toggle, run_button], justify="center"),
    ])
    return chaos_toggle, run_button, t_end_slider, theta1_slider, theta2_slider


@app.cell
def _(
    chaos_toggle,
    mo,
    np,
    run_button,
    t_end_slider,
    theta1_slider,
    theta2_slider,
    wasm_ready,
):
    mo.stop(not run_button.value)

    from simulations.physics.pendulum.params import PendulumParams
    from simulations.physics.pendulum.model import DoublePendulumSimulation

    _sim = DoublePendulumSimulation()
    _n_snapshots = 1200

    _params = PendulumParams(
        theta1_0=float(theta1_slider.value),
        theta2_0=float(theta2_slider.value),
        t_end=float(t_end_slider.value),
        n_snapshots=_n_snapshots,
    )

    with mo.status.spinner(title="Running simulation…"):
        result = _sim.run(_params)

        if chaos_toggle.value:
            _params2 = PendulumParams(
                theta1_0=float(theta1_slider.value) + 0.01,
                theta2_0=float(theta2_slider.value),
                t_end=float(t_end_slider.value),
                n_snapshots=_n_snapshots,
            )
            result2 = _sim.run(_params2)
            angular_separation = np.sqrt(
                (result.theta1 - result2.theta1) ** 2
                + (result.theta2 - result2.theta2) ** 2
            )
        else:
            result2 = None
            angular_separation = None
    return angular_separation, result, result2


@app.cell
def _(go, result, result2):
    """Bob 2 trajectory trace — the chaotic 'butterfly'."""
    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=result.x2,
        y=result.y2,
        mode="lines",
        line=dict(color="#1565C0", width=0.8),
        opacity=0.7,
        name="Trajectory",
    ))

    if result2 is not None:
        _fig.add_trace(go.Scatter(
            x=result2.x2,
            y=result2.y2,
            mode="lines",
            line=dict(color="#B71C1C", width=0.8),
            opacity=0.7,
            name="Perturbed (θ₁ + 0.01°)",
        ))

    # Pivot and initial bob positions as markers
    _fig.add_trace(go.Scatter(
        x=[0, result.x1[0], result.x2[0]],
        y=[0, result.y1[0], result.y2[0]],
        mode="markers+lines",
        marker=dict(size=[8, 8, 10], color=["black", "#1565C0", "#1565C0"]),
        line=dict(color="#1565C0", width=2, dash="dot"),
        name="Initial position",
        showlegend=True,
    ))

    _scale = result.params_snapshot["l1"] + result.params_snapshot["l2"]
    _fig.update_layout(
        title="Bob 2 trajectory trace",
        xaxis=dict(title="x (m)", scaleanchor="y", scaleratio=1,
                   range=[-_scale * 1.1, _scale * 1.1]),
        yaxis=dict(title="y (m)", range=[-_scale * 1.1, _scale * 1.1]),
        template="plotly_white",
        height=520,
        legend=dict(x=0.01, y=0.99),
    )
    _fig
    return


@app.cell
def _(go, np, result, result2):
    """Angles over time."""
    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=result.t, y=np.rad2deg(result.theta1),
        mode="lines", line=dict(color="#1565C0", width=1.2),
        name="θ₁",
    ))
    _fig.add_trace(go.Scatter(
        x=result.t, y=np.rad2deg(result.theta2),
        mode="lines", line=dict(color="#43A047", width=1.2),
        name="θ₂",
    ))
    if result2 is not None:
        _fig.add_trace(go.Scatter(
            x=result2.t, y=np.rad2deg(result2.theta1),
            mode="lines", line=dict(color="#B71C1C", width=1.0, dash="dash"),
            name="θ₁ perturbed", opacity=0.7,
        ))

    _fig.update_layout(
        title="Angles over time",
        xaxis_title="Time (s)",
        yaxis_title="Angle (°)",
        template="plotly_white",
        height=350,
        legend=dict(x=0.01, y=0.99),
    )
    _fig
    return


@app.cell
def _(angular_separation, go, make_subplots, result):
    """Energy conservation + chaos divergence."""
    _e0 = result.energy[0]
    _drift_pct = (result.energy - _e0) / abs(_e0) * 100.0

    if angular_separation is not None:
        _fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "Energy drift (%)",
            "Angular separation (rad)",
        ))
        _fig.add_trace(go.Scatter(
            x=result.t, y=_drift_pct,
            mode="lines", line=dict(color="#546E7A", width=1.2), name="ΔE/E₀",
        ), row=1, col=1)
        _fig.add_trace(go.Scatter(
            x=result.t, y=angular_separation,
            mode="lines", line=dict(color="#E65100", width=1.2), name="|Δθ|",
        ), row=1, col=2)
        _fig.update_xaxes(title_text="Time (s)")
        _fig.update_yaxes(title_text="ΔE/E₀ (%)", row=1, col=1)
        _fig.update_yaxes(title_text="√(Δθ₁² + Δθ₂²) (rad)", row=1, col=2)
        _fig.update_layout(
            template="plotly_white", height=320, showlegend=False,
            title="RK4 energy drift (left) and trajectory divergence from 0.01° perturbation (right)",
        )
    else:
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=result.t, y=_drift_pct,
            mode="lines", line=dict(color="#546E7A", width=1.2),
        ))
        _fig.update_layout(
            title="Energy drift (%)",
            xaxis_title="Time (s)",
            yaxis_title="ΔE / E₀ (%)",
            template="plotly_white",
            height=320,
        )

    _fig
    return


if __name__ == "__main__":
    app.run()
