import numpy as np

from simulations.casino.roulette.params import RouletteParams
from simulations.core.base import Simulation
from simulations.core.results import TimeSeriesResult

# Payout multipliers and win probabilities per bet type
_BET_CONFIG = {
    #               (payout, p_win_european, p_win_american)
    "red_black":    (1,      18 / 37,        18 / 38),
    "dozen":        (2,      12 / 37,        12 / 38),
    "single_number": (35,     1 / 37,          1 / 38),
}


class RouletteSimulation(Simulation[RouletteParams, TimeSeriesResult]):
    """Simulates player bankroll over N spins of roulette."""

    @property
    def name(self) -> str:
        return "Roulette"

    @property
    def description(self) -> str:
        return "Tracks bankroll over N spins with a fixed bet type and wheel."

    def run(self, params: RouletteParams) -> TimeSeriesResult:
        params.validate()
        rng = np.random.default_rng(params.seed)

        payout, p_european, p_american = _BET_CONFIG[params.bet_type]
        p_win = p_european if params.wheel_type == "european" else p_american

        rolls = rng.random(params.n_spins)
        wins = rolls < p_win
        # +payout*bet on win, -bet on loss
        outcomes = np.where(wins, params.bet_size * payout, -params.bet_size)

        running = params.initial_bankroll + np.cumsum(outcomes)

        # Find first bust (can't afford next bet) and zero from there
        bust = np.flatnonzero(running < params.bet_size)
        if bust.size:
            running[bust[0]:] = 0.0

        bankrolls = np.empty(params.n_spins + 1)
        bankrolls[0] = params.initial_bankroll
        bankrolls[1:] = running

        return TimeSeriesResult(
            simulation_name=self.name,
            params_snapshot=params.to_dict(),
            steps=np.arange(params.n_spins + 1),
            values=bankrolls,
        )
