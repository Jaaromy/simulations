"""
Tests for SIRSimulation.

Three required layers
---------------------
1. Determinism         — same params → bit-identical output.
2. Limit cases         — deterministic, analytically exact.
3. Theory cross-validation — N conservation, peak location, β=0 exponential decay.

Mathematical model
------------------
    dS/dt = −β S I / N
    dI/dt =  β S I / N  − γ I
    dR/dt =  γ I

    N = S + I + R  (conserved)
    R₀ = β / γ

Key analytical results used as test anchors
-------------------------------------------

β = 0 (no transmission):
    dI/dt = −γ I  →  I(t) = I₀ · exp(−γ t)
    S and R are constant.
    Source: standard ODE theory — independent of simulation.

R₀ < 1:
    When β S / N < γ for all t (guaranteed if S₀/N < 1/R₀ ≡ γ/β), then
    dI/dt < 0 everywhere → I is strictly decreasing from I₀.

R₀ > 1, epidemic peak:
    dI/dt = 0 when S = γN/β = N/R₀.
    Any β,γ,N where R₀ > 1 and S₀/N > 1/R₀ will produce a peak.
    At the peak, S must equal N/R₀ (within integration error).
    Source: standard SIR phase-plane analysis — independent of simulation.

N conservation:
    S(t) + I(t) + R(t) = N exactly for continuous dynamics.
    For RK4 at dt=0.1 this is preserved to within |ΔN|/N < 1 × 10⁻⁸.

β=0 exponential decay tolerance:
    I(t) = I₀ exp(−γt).  RK4 global error ∝ dt⁴.
    For dt=0.1, γ=0.1, t_end=50:  |ΔI/I| ≈ (γ·dt)⁴/30 · (γ·t_end) ≈ 3×10⁻⁷.
    Tolerance set to 0.01 % = 1×10⁻⁴ (3 orders of margin).
"""

import numpy as np
import pytest

from simulations.biology.sir.params import SIRParams
from simulations.biology.sir.model import SIRSimulation, SIRResult


@pytest.fixture
def sim() -> SIRSimulation:
    return SIRSimulation()


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------

def test_result_type(sim: SIRSimulation) -> None:
    result = sim.run(SIRParams(t_end=30.0))
    assert isinstance(result, SIRResult)


def test_output_length(sim: SIRSimulation) -> None:
    n = 100
    result = sim.run(SIRParams(t_end=30.0, n_snapshots=n))
    assert len(result.t) == n
    assert len(result.S) == n
    assert len(result.I) == n
    assert len(result.R) == n


def test_r0_property() -> None:
    assert abs(SIRParams(beta=0.3, gamma=0.05).R0 - 6.0) < 1e-10


def test_invalid_N_raises() -> None:
    with pytest.raises(ValueError, match="N"):
        SIRSimulation().run(SIRParams(N=0.0))


def test_invalid_I0_raises() -> None:
    with pytest.raises(ValueError, match="I0"):
        SIRSimulation().run(SIRParams(N=100.0, I0=200.0))


def test_invalid_beta_raises() -> None:
    with pytest.raises(ValueError, match="beta"):
        SIRSimulation().run(SIRParams(beta=-0.1))


def test_invalid_gamma_raises() -> None:
    with pytest.raises(ValueError, match="gamma"):
        SIRSimulation().run(SIRParams(gamma=-0.01))


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

def test_deterministic(sim: SIRSimulation) -> None:
    """SIR is a deterministic ODE: same params → bit-identical output."""
    params = SIRParams(N=1000.0, I0=5.0, beta=0.3, gamma=0.05, t_end=100.0)
    r1 = sim.run(params)
    r2 = sim.run(params)
    np.testing.assert_array_equal(r1.S, r2.S)
    np.testing.assert_array_equal(r1.I, r2.I)


# ---------------------------------------------------------------------------
# 2. Limit cases — deterministic, analytically exact
# ---------------------------------------------------------------------------

def test_beta_zero_no_new_infections(sim: SIRSimulation) -> None:
    """
    β = 0: no transmission force → S stays constant, R grows only from existing I.

    With β=0:
        dS/dt = 0  →  S(t) = S₀  (exactly)
        dI/dt = −γ I  →  I(t) = I₀ exp(−γt)
        dR/dt = γ I

    S must remain bit-stable; I must decay; R must increase monotonically.
    """
    params = SIRParams(N=1000.0, I0=50.0, beta=0.0, gamma=0.1, t_end=50.0,
                       dt=0.1, n_snapshots=200)
    result = sim.run(params)

    S0 = params.N - params.I0
    # S is exactly constant (no force of infection when β=0)
    np.testing.assert_allclose(result.S, S0, atol=1e-10,
                               err_msg="S must not change when β=0")
    # I decays monotonically
    assert (np.diff(result.I) <= 0).all(), "I must decrease monotonically when β=0"
    # R increases monotonically
    assert (np.diff(result.R) >= 0).all(), "R must increase monotonically when β=0"


def test_beta_zero_exponential_decay(sim: SIRSimulation) -> None:
    """
    β = 0: I(t) = I₀ exp(−γt).

    Derived tolerance (see module docstring):
        RK4 relative error < 0.01 % for dt=0.1, γ=0.1, t_end=50.

    Expected value from closed-form ODE solution — NOT from running the sim.
    """
    gamma = 0.1
    I0 = 100.0
    t_end = 50.0
    params = SIRParams(N=10_000.0, I0=I0, beta=0.0, gamma=gamma,
                       t_end=t_end, dt=0.1, n_snapshots=100)
    result = sim.run(params)

    I_theory = I0 * np.exp(-gamma * result.t)
    rel_err = np.abs(result.I - I_theory) / I0
    assert rel_err.max() < 1e-4, (
        f"β=0 exponential decay: max relative error {rel_err.max():.2e} > 0.01 %"
    )


def test_r0_below_one_no_epidemic(sim: SIRSimulation) -> None:
    """
    R₀ = β/γ ≤ 1 → dI/dt ≤ 0 always → I decreases monotonically from I₀.

    Proof: dI/dt = I(β S/N − γ).  With S ≤ N and β/γ < 1 we have β S/N < γ,
    so dI/dt < 0 for all I > 0.
    """
    params = SIRParams(N=10_000.0, I0=100.0, beta=0.05, gamma=0.1,
                       t_end=200.0, dt=0.1, n_snapshots=300)
    assert params.R0 < 1.0, f"Test setup error: R₀={params.R0} should be < 1"
    result = sim.run(params)
    assert (np.diff(result.I) <= 1e-9).all(), (
        f"R₀={params.R0:.2f} < 1: I must decrease monotonically"
    )


def test_no_infections_when_I0_zero(sim: SIRSimulation) -> None:
    """I₀ = 0 → no force of infection → S, I, R all constant."""
    params = SIRParams(N=5000.0, I0=0.0, beta=0.3, gamma=0.05, t_end=100.0, n_snapshots=50)
    result = sim.run(params)
    np.testing.assert_allclose(result.I, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.R, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.S, params.N, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Theory cross-validation
# ---------------------------------------------------------------------------

def test_population_conservation(sim: SIRSimulation) -> None:
    """
    S(t) + I(t) + R(t) = N must hold at every snapshot.

    For continuous dynamics this is exact.  RK4 with dt=0.1 preserves N
    to within |ΔN|/N < 1×10⁻⁸ (see module docstring).

    The expected value N is derived from the initial conditions — not from
    running the simulation.
    """
    N = 8_000.0
    params = SIRParams(N=N, I0=20.0, beta=0.3, gamma=0.05,
                       t_end=365.0, dt=0.1, n_snapshots=500)
    result = sim.run(params)
    total = result.S + result.I + result.R
    rel_err = np.abs(total - N) / N
    assert rel_err.max() < 1e-8, (
        f"Population not conserved: max |ΔN|/N = {rel_err.max():.2e} > 1e-8"
    )


def test_epidemic_peak_at_s_equals_n_over_r0() -> None:
    """
    Epidemic peak: dI/dt = 0 ↔ S = N/R₀ = γN/β.

    Derivation (independent of simulation):
        dI/dt = I(β S/N − γ) = 0  when S = γN/β.

    Protocol: find the snapshot with maximum I, check its S is within 3 %
    of N/R₀.  Tolerance accounts for discrete snapshot spacing and RK4 error.

    Expected S* = N/R₀ derived analytically — not measured from the run.
    """
    N = 10_000.0
    beta, gamma = 0.4, 0.1
    R0 = beta / gamma        # = 4.0
    S_star = N / R0          # = 2500.0  (from theory, NOT the simulation)

    params = SIRParams(N=N, I0=10.0, beta=beta, gamma=gamma,
                       t_end=200.0, dt=0.05, n_snapshots=2000)
    result = SIRSimulation().run(params)

    peak_idx = int(np.argmax(result.I))
    S_at_peak = result.S[peak_idx]

    rel_err = abs(S_at_peak - S_star) / S_star
    assert rel_err < 0.03, (
        f"Peak I at S={S_at_peak:.1f}; theory S*=N/R₀={S_star:.1f} "
        f"(rel error {rel_err:.3f} > 3 %)"
    )


def test_r0_greater_than_one_produces_epidemic(sim: SIRSimulation) -> None:
    """
    R₀ > 1 with S₀/N >> 1/R₀ → epidemic wave: I must exceed 10× initial value.

    This is not a tight quantitative bound; it is a gross qualitative check
    that the epidemic grows at all when theory predicts it should.
    """
    params = SIRParams(N=10_000.0, I0=10.0, beta=0.3, gamma=0.05,
                       t_end=365.0, dt=0.1, n_snapshots=500)
    assert params.R0 > 1.0
    result = sim.run(params)
    assert result.I.max() > 10 * params.I0, (
        f"R₀={params.R0}: epidemic expected but I never exceeded 10×I₀"
    )
