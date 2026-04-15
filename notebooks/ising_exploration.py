import marimo

__generated_with = "0.9.0"
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
    mo.md("# Ising Model — 2D Phase Transition")
    return


@app.cell
def _(mo):
    temperature = mo.ui.slider(
        start=0.5, stop=5.0, step=0.05, value=2.0,
        label='Temperature T <span class="tip" data-tip="In units where J=1, k_B=1. Critical temperature Tc ≈ 2.27. Below Tc: ordered (large domains). Above Tc: disordered (random noise).">?</span>',
    )
    grid_size = mo.ui.slider(
        start=20, stop=100, step=5, value=50,
        label='Grid Size L <span class="tip" data-tip="Lattice is L×L spins. Larger grids show cleaner domain structure and sharper phase transition, but run slower.">?</span>',
    )
    n_sweeps = mo.ui.slider(
        start=100, stop=5000, step=100, value=1000,
        label='Sweeps <span class="tip" data-tip="One sweep = L² Metropolis flip attempts. More sweeps give better equilibration and smoother magnetisation trajectory.">?</span>',
    )
    initial_state = mo.ui.dropdown(
        options=["random", "aligned_up", "aligned_down"],
        value="random",
        label='Initial state <span class="tip" data-tip="random: start disordered. aligned_up/down: start fully magnetised (faster equilibration in ordered phase).">?</span>',
    )
    mo.vstack([
        mo.hstack([temperature, grid_size], justify="center"),
        mo.hstack([n_sweeps, initial_state], justify="center"),
    ])
    return grid_size, initial_state, n_sweeps, temperature


@app.cell
def _(grid_size, initial_state, mo, n_sweeps, temperature, wasm_ready):
    import numpy as np
    from simulations.physics.ising.params import IsingParams
    from simulations.physics.ising.model import IsingSimulation

    _TC = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.2692

    params = IsingParams(
        grid_size=grid_size.value,
        temperature=temperature.value,
        n_sweeps=n_sweeps.value,
        n_snapshots=200,
        initial_state=initial_state.value,
        seed=0,
    )
    with mo.status.spinner(title="Running simulation…"):
        result = IsingSimulation().run(params)
    return np, params, result


@app.cell
def _(mo, np, params, result, temperature):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _TC = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.2692

    # --- Onsager prediction (only defined below Tc) ---
    _T = temperature.value
    if _T < _TC:
        _M_onsager = (1.0 - np.sinh(2.0 / _T) ** (-4)) ** 0.125
    else:
        _M_onsager = None

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=["Spin lattice (final state)", "Magnetisation |M| over sweeps"],
    )

    # --- Left: spin grid heatmap ---
    fig.add_trace(
        go.Heatmap(
            z=result.final_grid.tolist(),
            colorscale=[[0, "#1565C0"], [0.5, "#1565C0"], [0.5, "#B71C1C"], [1, "#B71C1C"]],
            zmin=-1, zmax=1,
            showscale=False,
        ),
        row=1, col=1,
    )

    # --- Right: magnetisation trajectory ---
    # Downsample to at most 500 points for display
    _snap_n = len(result.steps)
    _stride = max(1, _snap_n // 500)
    _x = result.steps[::_stride]
    _y = result.magnetization[::_stride]

    fig.add_trace(
        go.Scatter(
            x=_x, y=_y,
            mode="lines",
            line=dict(color="#5C6BC0", width=1.5),
            name="|M|",
        ),
        row=1, col=2,
    )

    # Onsager prediction line
    if _M_onsager is not None:
        fig.add_hline(
            y=_M_onsager,
            line_dash="dash",
            line_color="#F57F17",
            annotation_text=f"Onsager M = {_M_onsager:.3f}",
            annotation_position="top right",
            row=1, col=2,
        )

    # Critical temperature annotation
    _phase = "ordered" if _T < _TC else "disordered"
    _title = (
        f"T = {_T:.2f}  (Tc ≈ {_TC:.2f})  —  {_phase} phase  |  "
        f"final |M| = {result.magnetization[-1]:.3f}  |  "
        f"E/N = {result.energy_per_spin[-1]:.3f}"
    )

    fig.update_layout(
        title=_title,
        template="plotly_white",
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_xaxes(title_text="Sweep", row=1, col=2)
    fig.update_yaxes(title_text="|M|", range=[0, 1.05], row=1, col=2)

    mo.ui.plotly(fig)
    return fig, go, make_subplots


@app.cell
def _(mo, np, result, temperature):
    _TC = 2.0 / np.log(1.0 + np.sqrt(2.0))
    _T = temperature.value

    # Equilibrated mean: second half of trajectory
    _eq = result.magnetization[len(result.magnetization) // 2 :]
    _mean_mag = _eq.mean()
    _mean_e = result.energy_per_spin[len(result.energy_per_spin) // 2 :].mean()

    if _T < _TC:
        _M_onsager = (1.0 - np.sinh(2.0 / _T) ** (-4)) ** 0.125
        _deviation = abs(_mean_mag - _M_onsager)
        _onsager_row = f"| Onsager M_eq | {_M_onsager:.4f} |\n| Deviation | {_deviation:.4f} |"
    else:
        _onsager_row = "| Onsager M_eq | N/A (T > Tc) |"

    mo.md(f"""
### Equilibrated statistics (second half of run)

| Quantity | Value |
|---|---|
| Mean \\|M\\| | {_mean_mag:.4f} |
| Mean E/N | {_mean_e:.4f} |
{_onsager_row}
| Phase | {"**ordered**" if _T < _TC else "**disordered**"} |
""")
    return


if __name__ == "__main__":
    app.run()
