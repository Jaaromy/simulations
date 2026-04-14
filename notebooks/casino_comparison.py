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

    return go, mo, np


@app.cell
def _(mo):
    mo.md("""
    # Casino Simulation Comparison
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Parameters
    """)
    return


@app.cell
def _(mo):
    n_rounds = mo.ui.slider(
        start=100, stop=500_000, step=100, value=1_000,
        label='Rounds / Hands / Spins <abbr title="How many rounds each player plays per run. More rounds let the house edge grind the bankroll; short sessions show high variance.">[?]</abbr>',
    )
    initial_bankroll = mo.ui.number(
        start=100, stop=10000, step=100, value=1000,
        label='Initial Bankroll ($) <abbr title="Starting funds per simulated player (dashed line on chart). Smaller bankrolls ruin faster against the same house edge.">[?]</abbr>',
    )
    bet_size = mo.ui.number(
        start=1, stop=200, step=1, value=10,
        label='Bet Size ($) <abbr title="Flat bet per round. Expected loss per round = bet_size × house_edge. Larger bets amplify both wins and losses proportionally.">[?]</abbr>',
    )
    n_runs = mo.ui.slider(
        start=1, stop=50, step=1, value=10,
        label='Simulation Runs (per game) <abbr title="Independent player trajectories per game. More runs narrow the mean estimate and expose ruin probability.">[?]</abbr>',
    )
    use_basic_strategy = mo.ui.checkbox(
        value=True,
        label='Blackjack: Use Basic Strategy <abbr title="Optimal hit/stand/double/split for every hand. Cuts house edge from ~2% (guessing) to ~0.5%.">[?]</abbr>',
    )
    use_hilo = mo.ui.checkbox(
        value=False,
        label='Blackjack: Hi/Lo Card Counting (requires basic strategy) <abbr title="Track high vs low cards remaining. Raise bets when count favours the player. Requires basic strategy; can flip edge slightly positive at high counts.">[?]</abbr>',
    )
    hilo_max_bet_units = mo.ui.slider(
        start=2, stop=16, step=1, value=8,
        label='Hi/Lo Max Bet Units <abbr title="Bet ceiling as a multiple of base bet when count is strongly positive. Higher spread = more edge but more variance.">[?]</abbr>',
    )
    run_button = mo.ui.run_button(label="Run simulation")

    mo.vstack([
        mo.hstack([n_rounds, initial_bankroll, bet_size, n_runs], justify="start"),
        mo.hstack([use_basic_strategy, use_hilo, hilo_max_bet_units], justify="start"),
        run_button,
    ])
    return (
        bet_size,
        hilo_max_bet_units,
        initial_bankroll,
        n_rounds,
        n_runs,
        run_button,
        use_basic_strategy,
        use_hilo,
    )


@app.cell
def _(
    bet_size,
    hilo_max_bet_units,
    initial_bankroll,
    mo,
    n_rounds,
    n_runs,
    run_button,
    use_basic_strategy,
    use_hilo,
    wasm_ready,
):
    mo.stop(not run_button.value)
    from simulations.casino.blackjack.game import BlackjackSimulation
    from simulations.casino.blackjack.params import BlackjackParams
    from simulations.casino.roulette.game import RouletteSimulation
    from simulations.casino.roulette.params import RouletteParams
    from simulations.casino.slots.game import SlotsSimulation
    from simulations.casino.slots.params import SlotsParams

    _use_basic = use_basic_strategy.value or use_hilo.value  # Hi/Lo requires basic strategy
    bj_params = BlackjackParams(
        n_hands=n_rounds.value,
        initial_bankroll=initial_bankroll.value,
        bet_size=bet_size.value,
        use_basic_strategy=_use_basic,
        use_hilo=use_hilo.value,
        hilo_max_bet_units=hilo_max_bet_units.value,
    )
    rl_params = RouletteParams(
        n_spins=n_rounds.value,
        initial_bankroll=initial_bankroll.value,
        bet_size=bet_size.value,
    )
    sl_params = SlotsParams(
        n_spins=n_rounds.value,
        initial_bankroll=initial_bankroll.value,
        bet_size=bet_size.value,
    )

    _n = n_runs.value
    bj_runs, rl_runs, sl_runs = [], [], []

    _games = [
        ("Blackjack", BlackjackSimulation(), bj_params, bj_runs),
        ("Roulette",  RouletteSimulation(),  rl_params, rl_runs),
        ("Slots",     SlotsSimulation(),     sl_params, sl_runs),
    ]

    with mo.status.progress_bar(
        total=_n * len(_games),
        title="Running simulations…",
        completion_title="Done",
        show_rate=True,
        show_eta=True,
    ) as _bar:
        for _label, _sim, _params, _out in _games:
            for _ in range(_n):
                _out.append(_sim.run(_params))
                _bar.update(subtitle=_label)
    return bj_runs, rl_runs, sl_runs


@app.cell
def _(bj_runs, go, initial_bankroll, np, rl_runs, sl_runs):
    fig = go.Figure()

    _colors = {
        "Blackjack": "#2196F3",
        "Roulette":  "#F44336",
        "Slots":     "#4CAF50",
    }

    # Keep total embedded points bounded regardless of n_runs or n_rounds.
    # 30k points across all traces keeps the Plotly JSON well under 1 MB.
    _total_budget = 30_000
    _n_traces = len(bj_runs) + len(rl_runs) + len(sl_runs)
    _pts_per_trace = max(100, _total_budget // max(_n_traces, 1))

    for _label, _runs, _color in [
        ("Blackjack", bj_runs, _colors["Blackjack"]),
        ("Roulette",  rl_runs, _colors["Roulette"]),
        ("Slots",     sl_runs, _colors["Slots"]),
    ]:
        for _i, _run in enumerate(_runs):
            _stride = max(1, len(_run.steps) // _pts_per_trace)
            fig.add_trace(go.Scatter(
                x=_run.steps[::_stride],
                y=_run.values[::_stride],
                mode="lines",
                line=dict(color=_color, width=1),
                opacity=0.15,
                name=_label,
                legendgroup=_label,
                showlegend=(_i == 0),
            ))

        # Mean trajectory across all runs
        _mean = np.mean([r.values for r in _runs], axis=0)
        _stride = max(1, len(_runs[0].steps) // _pts_per_trace)
        fig.add_trace(go.Scatter(
            x=_runs[0].steps[::_stride],
            y=_mean[::_stride],
            mode="lines",
            line=dict(color=_color, width=2.5),
            opacity=1.0,
            name=f"{_label} (mean)",
            legendgroup=_label,
            showlegend=True,
        ))

    fig.add_hline(
        y=initial_bankroll.value,
        line_dash="dash",
        line_color="grey",
        annotation_text="Starting bankroll",
    )

    fig.update_layout(
        title="Bankroll over time — all games",
        xaxis_title="Round",
        yaxis_title="Bankroll ($)",
        legend_title="Game",
        template="plotly_white",
        height=500,
    )

    fig
    return


@app.cell
def _(bj_runs, go, np, rl_runs, sl_runs):
    summary_fig = go.Figure()

    for _label, _runs, _color in [
        ("Blackjack", bj_runs, "#2196F3"),
        ("Roulette",  rl_runs, "#F44336"),
        ("Slots",     sl_runs, "#4CAF50"),
    ]:
        _final = np.array([r.final_value for r in _runs])
        summary_fig.add_trace(go.Box(
            y=_final,
            name=_label,
            marker_color=_color,
            boxmean=True,
        ))

    summary_fig.update_layout(
        title="Final bankroll distribution by game",
        yaxis_title="Final Bankroll ($)",
        template="plotly_white",
        height=400,
    )

    summary_fig
    return


if __name__ == "__main__":
    app.run()
