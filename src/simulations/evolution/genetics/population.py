import numpy as np

from simulations.core.base import Simulation
from simulations.core.results import TimeSeriesResult
from simulations.evolution.genetics.params import GeneticsParams


class GeneticsSimulation(Simulation[GeneticsParams, TimeSeriesResult]):
    """Simulates population size over N generations using a logistic growth model.

    Growth is density-dependent: as population approaches carrying_capacity,
    effective birth rate falls and effective death rate rises, producing
    the classic S-curve with stochastic noise.
    """

    @property
    def name(self) -> str:
        return "Genetics"

    @property
    def description(self) -> str:
        return "Logistic population growth with stochastic birth/death events."

    def run(self, params: GeneticsParams) -> TimeSeriesResult:
        params.validate()
        rng = np.random.default_rng(params.seed)

        population = float(params.initial_population)
        populations = np.empty(params.n_generations + 1)
        populations[0] = population

        for i in range(params.n_generations):
            if population <= 0:
                populations[i + 1 :] = 0.0
                break

            # Logistic scaling: growth slows as population approaches K
            density = population / params.carrying_capacity
            effective_birth = params.birth_rate
            effective_death = params.death_rate + (params.birth_rate - params.death_rate) * density

            births = rng.poisson(max(effective_birth * population, 0))
            deaths = rng.poisson(max(effective_death * population, 0))

            population = max(population + births - deaths, 0.0)
            populations[i + 1] = population

        return TimeSeriesResult(
            simulation_name=self.name,
            params_snapshot=params.to_dict(),
            steps=np.arange(params.n_generations + 1),
            values=populations,
        )
