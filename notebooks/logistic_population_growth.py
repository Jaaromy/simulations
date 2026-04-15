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
    mo.md("# Logistic Population Growth")
    return


@app.cell
def _(mo):
    n_generations = mo.ui.slider(
        start=10, stop=500, step=10, value=100,
        label='Generations <span class="tip" data-tip="Total discrete time steps. Each generation applies births, deaths, and density-dependent regulation once.">?</span>',
    )
    initial_population = mo.ui.number(
        start=10, stop=10000, step=10, value=100,
        label='Initial Population <span class="tip" data-tip="Population size at generation 0. Start far below carrying capacity to see the full S-curve approach to equilibrium.">?</span>',
    )
    birth_rate = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.15,
        label='Birth Rate <span class="tip" data-tip="Per-individual probability of producing one offspring per generation. Net intrinsic growth rate r = birth_rate − death_rate.">?</span>',
    )
    death_rate = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.05,
        label='Death Rate <span class="tip" data-tip="Per-individual probability of dying per generation. When death_rate ≥ birth_rate the population declines to zero regardless of K.">?</span>',
    )
    carrying_capacity = mo.ui.number(
        start=100, stop=50000, step=100, value=5000,
        label='Carrying Capacity <span class="tip" data-tip="Maximum sustainable population K. Equilibrium N* = K exactly. Density-dependent feedback raises effective death rate as N→K.">?</span>',
    )
    n_runs = mo.ui.slider(
        start=1, stop=20, step=1, value=5,
        label='Simulation Runs <span class="tip" data-tip="Independent stochastic trajectories. More runs reveal outcome spread and stabilise the mean.">?</span>',
    )

    mo.vstack([
        mo.hstack([n_generations, initial_population, n_runs], justify="center"),
        mo.hstack([birth_rate, death_rate, carrying_capacity], justify="center"),
    ])
    return birth_rate, carrying_capacity, death_rate, initial_population, n_generations, n_runs


@app.cell
def _(birth_rate, carrying_capacity, death_rate, initial_population, n_generations, mo, n_runs, wasm_ready):
    from simulations.evolution.logistic.params import LogisticParams
    from simulations.evolution.logistic.population import LogisticSimulation

    params = LogisticParams(
        n_generations=n_generations.value,
        initial_population=int(initial_population.value),
        birth_rate=birth_rate.value,
        death_rate=death_rate.value,
        carrying_capacity=int(carrying_capacity.value),
    )

    with mo.status.spinner(title="Running simulation…"):
        runs = LogisticSimulation().run_batch(params, n_runs.value)
    return params, runs


@app.cell
def _(runs, params):
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()

    _max_points = 2000
    _stride = max(1, len(runs[0].steps) // _max_points)
    for _i, _run in enumerate(runs):
        fig.add_trace(go.Scatter(
            x=_run.steps[::_stride],
            y=_run.values[::_stride],
            mode="lines",
            line=dict(color="#9C27B0", width=1),
            opacity=0.5,
            name="Run",
            legendgroup="runs",
            showlegend=(_i == 0),
        ))

    # Mean trajectory
    _mean = np.mean(np.stack([r.values for r in runs]), axis=0)
    fig.add_trace(go.Scatter(
        x=runs[0].steps[::_stride],
        y=_mean[::_stride],
        mode="lines",
        line=dict(color="#4A148C", width=3),
        name="Mean",
    ))

    fig.add_hline(
        y=params.carrying_capacity,
        line_dash="dash",
        line_color="grey",
        annotation_text="Carrying capacity",
    )

    fig.update_layout(
        title="Population size over generations",
        xaxis_title="Generation",
        yaxis_title="Population",
        template="plotly_white",
        height=500,
    )

    fig
    return (fig,)


if __name__ == "__main__":
    app.run()
