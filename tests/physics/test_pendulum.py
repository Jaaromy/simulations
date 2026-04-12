"""
Tests for DoublePendulumSimulation.

Three required layers
---------------------
1. Seed reproducibility  — deterministic: same params → bit-identical output.
2. Limit cases           — deterministic bounds with exactly known answers.
3. Theory cross-validation — energy conservation and small-angle normal modes.

Mathematical model
------------------
State: [θ₁, ω₁, θ₂, ω₂]
EOMs (exact Lagrangian, no small-angle approx):

    D = 2m₁ + m₂ − m₂ cos(2Δ),  Δ = θ₁ − θ₂

    θ̈₁ = [−g(2m₁+m₂)sinθ₁ − m₂g sin(θ₁−2θ₂)
           − 2 sinΔ · m₂(ω₂²l₂ + ω₁²l₁ cosΔ)] / (l₁ D)

    θ̈₂ = [2 sinΔ (ω₁²l₁(m₁+m₂) + g(m₁+m₂)cosθ₁ + ω₂²l₂ m₂ cosΔ)] / (l₂ D)

Conserved quantity: total mechanical energy
    T = ½(m₁+m₂)l₁²ω₁² + ½m₂l₂²ω₂² + m₂l₁l₂ω₁ω₂cos(Δ)
    V = −(m₁+m₂)g l₁ cosθ₁ − m₂g l₂ cosθ₂
    E = T + V   (constant for exact dynamics)

Small-angle normal modes (m₁=m₂=m, l₁=l₂=l)
----------------------------------------------
Linearising around (0,0):
    ω²± = (g/l)(2 ± √2)

Lower mode (−): eigenvector θ₂/θ₁ = +√2.
For m=1, l=1, g=9.81:
    ω₋ = sqrt(9.81 × (2−√2)) ≈ 2.397 rad/s
    T₋ = 2π / ω₋ ≈ 2.622 s

Starting purely in the lower mode (θ₁=ε, θ₂=√2 ε, ω₁=ω₂=0) and integrating
for exactly T₋ must return θ₁ ≈ ε (within 0.5 % for ε = 0.01 rad).
"""

import numpy as np
import pytest

from simulations.physics.pendulum.params import PendulumParams
from simulations.physics.pendulum.model import DoublePendulumSimulation, PendulumResult


@pytest.fixture
def sim() -> DoublePendulumSimulation:
    return DoublePendulumSimulation()


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------

def test_result_type(sim: DoublePendulumSimulation) -> None:
    result = sim.run(PendulumParams(t_end=1.0, n_snapshots=50))
    assert isinstance(result, PendulumResult)


def test_output_length(sim: DoublePendulumSimulation) -> None:
    n = 200
    result = sim.run(PendulumParams(t_end=2.0, n_snapshots=n))
    assert len(result.t) == n
    assert len(result.theta1) == n
    assert len(result.x2) == n
    assert len(result.energy) == n


def test_time_starts_at_zero_ends_near_t_end(sim: DoublePendulumSimulation) -> None:
    t_end = 5.0
    result = sim.run(PendulumParams(t_end=t_end, n_snapshots=100))
    assert result.t[0] == 0.0
    assert abs(result.t[-1] - t_end) < 0.01


def test_invalid_mass_raises() -> None:
    with pytest.raises(ValueError, match="m1"):
        DoublePendulumSimulation().run(PendulumParams(m1=0.0))


def test_invalid_length_raises() -> None:
    with pytest.raises(ValueError, match="l2"):
        DoublePendulumSimulation().run(PendulumParams(l2=-1.0))


def test_invalid_g_raises() -> None:
    with pytest.raises(ValueError, match="g"):
        DoublePendulumSimulation().run(PendulumParams(g=-1.0))


# ---------------------------------------------------------------------------
# 1. Determinism — same params → bit-identical output
# ---------------------------------------------------------------------------

def test_deterministic(sim: DoublePendulumSimulation) -> None:
    """Double pendulum is a deterministic ODE: same params must give identical results."""
    params = PendulumParams(theta1_0=120.0, theta2_0=-20.0, t_end=5.0, n_snapshots=100)
    r1 = sim.run(params)
    r2 = sim.run(params)
    np.testing.assert_array_equal(r1.theta1, r2.theta1)
    np.testing.assert_array_equal(r1.x2, r2.x2)
    np.testing.assert_array_equal(r1.energy, r2.energy)


# ---------------------------------------------------------------------------
# 2. Limit cases — deterministic, no statistics
# ---------------------------------------------------------------------------

def test_g_zero_frozen(sim: DoublePendulumSimulation) -> None:
    """
    g = 0, ω₁ = ω₂ = 0 → no torques, no velocities → system frozen forever.

    All θ̈ terms in the EOM that survive when g=0 and ω=0:
        num₁ = −2 sinΔ · m₂ (0·l₂ + 0·l₁·cosΔ) = 0
        num₂ = 2 sinΔ (0 + 0 + 0) = 0
    So θ̈₁ = θ̈₂ = 0, and angles never change.

    Deterministic: no tolerance bands.
    """
    th1_0, th2_0 = 45.0, -60.0
    params = PendulumParams(
        theta1_0=th1_0,
        theta2_0=th2_0,
        omega1_0=0.0,
        omega2_0=0.0,
        g=0.0,
        t_end=10.0,
        dt=0.01,
        n_snapshots=50,
    )
    result = sim.run(params)
    np.testing.assert_allclose(result.theta1, np.deg2rad(th1_0), atol=1e-12)
    np.testing.assert_allclose(result.theta2, np.deg2rad(th2_0), atol=1e-12)
    np.testing.assert_allclose(result.omega1, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.omega2, 0.0, atol=1e-12)


def test_initial_energy_correct() -> None:
    """
    Manual energy check at t=0 with known IC:
        θ₁ = π/2, θ₂ = 0, ω₁ = ω₂ = 0, m₁=m₂=1, l₁=l₂=1, g=9.81

    T = 0 (no velocities)
    V = −(1+1)·9.81·1·cos(π/2) − 1·9.81·1·cos(0)
      = −2·9.81·0 − 9.81·1 = −9.81

    E₀ = −9.81

    Any wrong sign in cosθ₁ or cosθ₂ term will give a different number.
    """
    params = PendulumParams(
        theta1_0=90.0,   # π/2 rad
        theta2_0=0.0,
        omega1_0=0.0,
        omega2_0=0.0,
        m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81,
        t_end=0.01, dt=0.001, n_snapshots=2,
    )
    result = DoublePendulumSimulation().run(params)
    expected = -9.81   # derived above — does NOT come from running the sim
    assert abs(result.energy[0] - expected) < 1e-9, (
        f"Initial energy: expected {expected}, got {result.energy[0]}"
    )


def test_positions_from_angles() -> None:
    """
    Cartesian positions at t=0:
        θ₁ = 90°  → (x₁, y₁) = (l₁·sin90°, −l₁·cos90°) = (l₁, 0)
        θ₂ = −90° → bob 2 is directly left of bob 1:
            x₂ = x₁ + l₂·sin(−90°) = l₁ − l₂
            y₂ = y₁ − l₂·cos(−90°) = 0

    For l₁=2, l₂=1:  x₁=2, y₁=0, x₂=1, y₂=0.
    Confirms sign conventions in the position formulae.
    g=0 so the system stays frozen (confirmed by test_g_zero_frozen).
    """
    params = PendulumParams(
        theta1_0=90.0,
        theta2_0=-90.0,
        omega1_0=0.0, omega2_0=0.0,
        l1=2.0, l2=1.0, g=0.0,
        t_end=0.01, dt=0.001, n_snapshots=2,
    )
    result = DoublePendulumSimulation().run(params)
    np.testing.assert_allclose(result.x1[0], 2.0, atol=1e-10)
    np.testing.assert_allclose(result.y1[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(result.x2[0], 1.0, atol=1e-10)
    np.testing.assert_allclose(result.y2[0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Theory cross-validation
# ---------------------------------------------------------------------------

def test_energy_conservation() -> None:
    """
    Total mechanical energy E = T + V is exactly conserved by the continuous EOMs.
    For fixed-step RK4 at dt=0.001 s, 10 s, moderate IC (45°/−30°):

    Derived tolerance:
        RK4 local truncation error ∝ dt⁵; global error ∝ dt⁴.
        Empirical for this system at these settings: |ΔE/E₀| < 0.01 %.
        We use 0.1 % as a conservative upper bound.

    Expected value (E₀) is computed from the SAME initial conditions at t=0,
    not from a prior run.  E₀ is a function of ICs and physical parameters only.
    """
    params = PendulumParams(
        theta1_0=45.0,
        theta2_0=-30.0,
        omega1_0=0.0,
        omega2_0=0.0,
        m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81,
        t_end=10.0,
        dt=0.001,
        n_snapshots=500,
    )
    result = DoublePendulumSimulation().run(params)
    E0 = result.energy[0]
    drift = np.abs(result.energy - E0) / np.abs(E0)
    max_drift = drift.max()
    assert max_drift < 1e-3, (
        f"Energy drift {max_drift:.2e} exceeds 0.1 % over 10 s at dt=0.001"
    )


def test_small_angle_lower_normal_mode() -> None:
    """
    Small-angle normal modes for m₁=m₂=1, l₁=l₂=1, g=9.81 (derived, not measured):

        ω²± = g(2 ± √2) / l

    Lower mode (−): eigenvector θ₂/θ₁ = +√2.
    Exciting only the lower mode: θ₁₀ = ε = 0.01 rad, θ₂₀ = √2·ε, ω₁₀=ω₂₀=0.

    After exactly one period T₋ = 2π/ω₋, the system must return to the same
    configuration: |θ₁(T₋) − θ₁₀| / θ₁₀ < 0.005  (0.5 %).

    Tolerance derivation:
        - Small-angle approx error: O(ε²) ≈ 10⁻⁴  (negligible at ε=0.01)
        - RK4 global error at dt=0.001 over ~2.6 s: O(dt⁴·T) ≈ 2.6×10⁻¹²  (negligible)
        - Mode coupling at ε=0.01: < 0.1 % of amplitude
        Chosen tolerance: 0.5 % provides large margin over all error sources.

    Expected value derived analytically from published normal-mode theory,
    not from measuring a simulation run.
    """
    g, l = 9.81, 1.0
    omega_minus = np.sqrt(g * (2.0 - np.sqrt(2)) / l)  # ≈ 2.397 rad/s
    T_minus = 2.0 * np.pi / omega_minus                 # ≈ 2.622 s

    eps = 0.01  # small-angle amplitude (rad)
    params = PendulumParams(
        theta1_0=np.rad2deg(eps),
        theta2_0=np.rad2deg(np.sqrt(2) * eps),
        omega1_0=0.0,
        omega2_0=0.0,
        m1=1.0, m2=1.0, l1=l, l2=l, g=g,
        t_end=T_minus,
        dt=0.001,
        n_snapshots=int(T_minus / 0.001) + 1,  # keep every integration step
    )
    result = DoublePendulumSimulation().run(params)

    # θ₁ at t ≈ T₋ should be back near eps (one full oscillation)
    theta1_final = result.theta1[-1]
    rel_error = abs(theta1_final - eps) / eps
    assert rel_error < 0.005, (
        f"Lower normal mode: expected θ₁(T₋) ≈ {eps:.4f} rad, "
        f"got {theta1_final:.6f} (rel error {rel_error:.4f} > 0.5 %)"
    )
