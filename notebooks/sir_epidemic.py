import marimo

__generated_with = "0.23.1"
app = marimo.App()


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

    return go, mo, np


@app.cell
def _(mo):
    mo.md("""
    # SIR Epidemic Model
    """)
    return


@app.cell
def _(mo):
    N_input = mo.ui.number(
        start=100, stop=1_000_000, step=100, value=10_000,
        label='Population N <abbr title="Total number of individuals. Divided into S+I+R = N at all times.">[?]</abbr>',
    )
    I0_input = mo.ui.number(
        start=1, stop=10_000, step=1, value=10,
        label='Initial infected I₀ <abbr title="Number of infectious individuals at t=0. All others start as susceptible.">[?]</abbr>',
    )
    beta_slider = mo.ui.slider(
        start=0.01, stop=1.0, step=0.01, value=0.3,
        label='β — transmission rate (day⁻¹) <abbr title="Rate at which a susceptible becomes infected per infectious contact. β = contact rate × transmission probability.">[?]</abbr>',
    )
    gamma_slider = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.05,
        label='γ — recovery rate (day⁻¹) <abbr title="Fraction of infectious who recover per day. Mean infectious period = 1/γ days.">[?]</abbr>',
    )
    t_end_input = mo.ui.slider(
        start=30, stop=730, step=10, value=365,
        label='Duration (days) <abbr title="Simulation horizon in days.">[?]</abbr>',
    )
    run_button = mo.ui.run_button(label="Run simulation")

    mo.vstack([
        mo.hstack([N_input, I0_input], justify="start"),
        mo.hstack([beta_slider, gamma_slider, t_end_input], justify="start"),
        run_button,
    ])
    return (
        I0_input,
        N_input,
        beta_slider,
        gamma_slider,
        run_button,
        t_end_input,
    )


@app.cell
def _(
    I0_input,
    N_input,
    beta_slider,
    gamma_slider,
    mo,
    run_button,
    t_end_input,
    wasm_ready,
):
    mo.stop(not run_button.value)

    from simulations.biology.sir.params import SIRParams
    from simulations.biology.sir.model import SIRSimulation

    _params = SIRParams(
        N=float(N_input.value),
        I0=float(I0_input.value),
        beta=float(beta_slider.value),
        gamma=float(gamma_slider.value),
        t_end=float(t_end_input.value),
        dt=0.1,
        n_snapshots=600,
    )

    result = SIRSimulation().run(_params)
    return (result,)


@app.cell
def _(mo, result):
    """Key metrics callout row."""
    _R0 = result.R0
    _N = result.params_snapshot["N"]
    _herd_pct = max(0.0, (1.0 - 1.0 / _R0) * 100.0) if _R0 > 1.0 else 0.0
    _peak_I = result.I.max()
    _peak_pct = _peak_I / _N * 100.0
    _final_R_pct = result.R[-1] / _N * 100.0

    mo.hstack([
        mo.stat(
            label="Basic reproduction number R₀",
            value=f"{_R0:.2f}",
            caption="β / γ  — epidemic grows when R₀ > 1",
            bordered=True,
        ),
        mo.stat(
            label="Herd immunity threshold",
            value=f"{_herd_pct:.1f} %",
            caption="1 − 1/R₀  of population needs immunity to halt spread",
            bordered=True,
        ),
        mo.stat(
            label="Peak infected",
            value=f"{_peak_I:,.0f}  ({_peak_pct:.1f} %)",
            caption="Maximum simultaneous infectious individuals",
            bordered=True,
        ),
        mo.stat(
            label="Total infected (final)",
            value=f"{result.R[-1]:,.0f}  ({_final_R_pct:.1f} %)",
            caption="Cumulative cases = R(∞)  (all who caught the disease)",
            bordered=True,
        ),
    ], justify="start", gap="1rem")
    return


@app.cell
def _(go, result):
    """S / I / R curves over time."""
    _N = result.params_snapshot["N"]
    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=result.t, y=result.S / _N * 100,
        mode="lines", line=dict(color="#1565C0", width=2),
        name="Susceptible (S)",
    ))
    _fig.add_trace(go.Scatter(
        x=result.t, y=result.I / _N * 100,
        mode="lines", line=dict(color="#B71C1C", width=2),
        name="Infectious (I)",
        fill="tozeroy", fillcolor="rgba(183,28,28,0.08)",
    ))
    _fig.add_trace(go.Scatter(
        x=result.t, y=result.R / _N * 100,
        mode="lines", line=dict(color="#2E7D32", width=2),
        name="Recovered (R)",
    ))

    # Vertical line at epidemic peak
    _peak_t = result.t[int(result.I.argmax())]
    _fig.add_vline(
        x=_peak_t,
        line_dash="dash", line_color="grey",
        annotation_text=f"Peak (day {_peak_t:.0f})",
        annotation_position="top right",
    )

    _fig.update_layout(
        title="Epidemic dynamics — S / I / R over time",
        xaxis_title="Time (days)",
        yaxis_title="Population (%)",
        template="plotly_white",
        height=420,
        legend=dict(x=0.7, y=0.9),
    )
    _fig
    return


@app.cell
def _(go, np, result):
    """Phase portrait: S vs I (the SIR trajectory in state space)."""
    _N = result.params_snapshot["N"]
    _R0 = result.R0

    _fig = go.Figure()

    _fig.add_trace(go.Scatter(
        x=result.S / _N,
        y=result.I / _N,
        mode="lines",
        line=dict(color="#6A1B9A", width=1.5),
        name="Trajectory",
    ))

    # Mark start and end
    _fig.add_trace(go.Scatter(
        x=[result.S[0] / _N, result.S[-1] / _N],
        y=[result.I[0] / _N, result.I[-1] / _N],
        mode="markers",
        marker=dict(size=10, color=["#1565C0", "#B71C1C"],
                    symbol=["circle", "x"]),
        name="Start / End",
    ))

    # Threshold line S = 1/R₀ (peak location)
    if _R0 > 1.0:
        _s_star = 1.0 / _R0
        _fig.add_vline(
            x=_s_star,
            line_dash="dot", line_color="grey",
            annotation_text=f"S* = 1/R₀ = {_s_star:.3f}",
            annotation_position="top right",
        )

    # Analytical phase-curve overlay: I = −S + (1/R₀)·ln(S) + C, normalised by N
    _S0_n = result.S[0] / _N
    _I0_n = result.I[0] / _N
    if _R0 > 1.0:
        _s_vals = np.linspace(result.S[-1] / _N, _S0_n, 300)
        _C = _I0_n + _S0_n - (1.0 / _R0) * np.log(_S0_n)
        _I_curve = -_s_vals + (1.0 / _R0) * np.log(_s_vals) + _C
        _I_curve = np.clip(_I_curve, 0, None)
        _fig.add_trace(go.Scatter(
            x=_s_vals, y=_I_curve,
            mode="lines",
            line=dict(color="#E65100", width=1.2, dash="dash"),
            name="Analytical phase curve",
        ))

    _fig.update_layout(
        title="Phase portrait: S vs I",
        xaxis_title="Susceptible fraction S/N",
        yaxis_title="Infectious fraction I/N",
        template="plotly_white",
        height=400,
        legend=dict(x=0.6, y=0.9),
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
