import numpy as np

from simulations.casino.slots.params import SlotsParams
from simulations.core.base import Simulation
from simulations.core.results import TimeSeriesResult


_WIN_PROBABILITY = 0.15  # Fraction of spins that land a win; must match exponential scale below


class SlotsSimulation(Simulation[SlotsParams, TimeSeriesResult]):
    """Simulates player bankroll over N spins of a slot machine.

    Uses a simplified model: each spin independently returns the bet
    multiplied by a random payout drawn from an exponential distribution
    scaled so the expected value equals the RTP. This produces realistic
    variance (mostly small losses, occasional large wins) without needing
    to model a full reel strip.
    """

    @property
    def name(self) -> str:
        return "Slots"

    @property
    def description(self) -> str:
        return "Tracks bankroll over N spins with configurable RTP."

    def run(self, params: SlotsParams) -> TimeSeriesResult:
        params.validate()
        rng = np.random.default_rng(params.seed)

        # ~85% of spins lose; ~15% win with exponential payout
        # Scaled so E[payout] = rtp * bet_size
        win_mask = rng.random(params.n_spins) < _WIN_PROBABILITY
        # Exponential payouts for winning spins; scale = rtp / WIN_PROBABILITY
        payouts = rng.exponential(scale=params.rtp / _WIN_PROBABILITY, size=params.n_spins)

        # Net per spin: win → payout*bet - bet; lose → -bet
        outcomes = np.where(win_mask, (payouts - 1.0) * params.bet_size, -params.bet_size)

        running = params.initial_bankroll + np.cumsum(outcomes)

        # Find first bust and zero from there
        bust = np.flatnonzero(running < params.bet_size)
        if bust.size:
            running[bust[0]:] = 0.0

        bankrolls = np.empty(params.n_spins + 1)
        bankrolls[0] = params.initial_bankroll
        bankrolls[1:] = np.maximum(running, 0.0)

        return TimeSeriesResult(
            simulation_name=self.name,
            params_snapshot=params.to_dict(),
            steps=np.arange(params.n_spins + 1),
            values=bankrolls,
        )
