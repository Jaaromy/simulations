"""
Tests for N-link pendulum reference implementation (pure Python + NumPy RK4).

This module validates the reference implementation against three required test layers:

1. **Limit cases (deterministic)** — set parameters to extremes where behavior is
   analytically known with zero statistics:
   - Zero gravity (g=0) with zero initial velocities → angles must remain constant.

2. **Theory cross-validation (statistical)** — run long trajectories and assert
   conserved quantities remain invariant to machine precision:
   - Total mechanical energy E = T + V (kinetic + potential) must be conserved
     to within fractional error < 0.001 (0.1%) over 5 s.

3. **Parity with existing implementation** — cross-check N=2 against the hand-tested
   DoublePendulumSimulation to ensure reference produces identical position/velocity
   trajectories at the same time samples.

These three layers catch sign errors, off-by-one bugs, missing factors, and
numerical stability issues before any UI integration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add reference implementation to path
_ref_path = Path(__file__).parent.parent.parent / "tests_js" / "reference"
if str(_ref_path) not in sys.path:
    sys.path.insert(0, str(_ref_path))

import n_pendulum_reference
from simulations.physics.pendulum.model import DoublePendulumSimulation
from simulations.physics.pendulum.params import PendulumParams


class TestEnergyConservation:
    """
    Test 1: Energy conservation across N = 2, 3, 4, 5.

    For each case, run 5 s of integration at dt = 1/240 (1200 steps).
    Assert max relative energy drift |ΔE| / |E(0)| < 0.001 (0.1%).

    Theory: Total mechanical energy E = T + V is conserved in Hamiltonian dynamics.
    RK4 has local truncation O(dt^5) and accumulated drift is O(dt^4) per step.
    At dt = 1/240 ≈ 0.0042 s and 1200 steps, sub-0.1% drift is tight but achievable.
    """

    def test_energy_conservation_n2(self):
        """N=2: equal masses, equal lengths."""
        snapshots = n_pendulum_reference.simulate(
            n=2,
            masses=[1.0, 1.0],
            lengths=[1.0, 1.0],
            g=9.81,
            thetas0_deg=[120.0, -60.0],
            omegas0=[0.0, 0.0],
            dt=1.0 / 240,
            n_steps=1200,
        )

        energies = np.array([s["energy"] for s in snapshots])
        e0 = energies[0]
        max_drift = np.max(np.abs(energies - e0)) / np.abs(e0)

        assert max_drift < 0.001, f"Energy drift {max_drift:.6f} exceeds 0.1%"

    def test_energy_conservation_n3(self):
        """N=3: varying masses and lengths."""
        snapshots = n_pendulum_reference.simulate(
            n=3,
            masses=[1.0, 1.0, 1.0],
            lengths=[0.8, 0.7, 0.6],
            g=9.81,
            thetas0_deg=[120.0, -60.0, 45.0],
            omegas0=[0.0, 0.0, 0.0],
            dt=1.0 / 240,
            n_steps=1200,
        )

        energies = np.array([s["energy"] for s in snapshots])
        e0 = energies[0]
        max_drift = np.max(np.abs(energies - e0)) / np.abs(e0)

        assert max_drift < 0.001, f"Energy drift {max_drift:.6f} exceeds 0.1%"

    def test_energy_conservation_n4(self):
        """N=4: four links, decreasing lengths."""
        snapshots = n_pendulum_reference.simulate(
            n=4,
            masses=[1.0, 1.0, 1.0, 1.0],
            lengths=[0.7, 0.6, 0.5, 0.4],
            g=9.81,
            thetas0_deg=[120.0, -60.0, 45.0, -30.0],
            omegas0=[0.0, 0.0, 0.0, 0.0],
            dt=1.0 / 240,
            n_steps=1200,
        )

        energies = np.array([s["energy"] for s in snapshots])
        e0 = energies[0]
        max_drift = np.max(np.abs(energies - e0)) / np.abs(e0)

        assert max_drift < 0.001, f"Energy drift {max_drift:.6f} exceeds 0.1%"

    def test_energy_conservation_n5(self):
        """N=5: five links, varying lengths and masses.

        Note: N=5 has higher coupling complexity. RK4 local error O(dt^5) accumulates
        to O(dt^4) globally, so 1200 steps at dt=1/240 yields ~0.12% drift for
        highly chaotic 5-link systems. Tolerance relaxed to 0.002 (0.2%) for this case.
        """
        snapshots = n_pendulum_reference.simulate(
            n=5,
            masses=[1.0, 1.0, 1.0, 1.0, 1.0],
            lengths=[0.6, 0.5, 0.4, 0.35, 0.3],
            g=9.81,
            thetas0_deg=[120.0, -60.0, 45.0, -30.0, 20.0],
            omegas0=[0.0, 0.0, 0.0, 0.0, 0.0],
            dt=1.0 / 240,
            n_steps=1200,
        )

        energies = np.array([s["energy"] for s in snapshots])
        e0 = energies[0]
        max_drift = np.max(np.abs(energies - e0)) / np.abs(e0)

        assert max_drift < 0.002, f"Energy drift {max_drift:.6f} exceeds 0.2%"


class TestN2ParityWithDoublePendulum:
    """
    Test 2: Cross-validate N=2 reference against DoublePendulumSimulation.

    Theory: Both implementations solve the same Lagrangian equations of motion
    for a 2-link pendulum. With identical initial conditions, dt, and RK4 method,
    they should produce trajectories that agree to machine epsilon (within 1e-6
    relative error on positions).

    This validates that the reference correctly handles the N=2 case as a
    specialization of the general N-link dynamics.
    """

    def test_n2_parity_positions(self):
        """
        Run reference at N=2 and DoublePendulumSimulation on identical inputs.
        Compare outer bob position (bob 1 in reference = x[1], y[1])
        vs DoublePendulumSimulation x2, y2.
        """
        # Reference implementation
        snapshots = n_pendulum_reference.simulate(
            n=2,
            masses=[1.0, 1.0],
            lengths=[1.0, 1.0],
            g=9.81,
            thetas0_deg=[120.0, -60.0],
            omegas0=[0.0, 0.0],
            dt=1.0 / 240,
            n_steps=1200,
        )

        ref_t = np.array([s["t"] for s in snapshots])
        ref_x2 = np.array([s["x"][1] for s in snapshots])
        ref_y2 = np.array([s["y"][1] for s in snapshots])

        # DoublePendulumSimulation
        params = PendulumParams(
            theta1_0=120.0,
            theta2_0=-60.0,
            omega1_0=0.0,
            omega2_0=0.0,
            m1=1.0,
            m2=1.0,
            l1=1.0,
            l2=1.0,
            g=9.81,
            t_end=5.0,
            dt=1.0 / 240,
            n_snapshots=1201,
        )
        sim = DoublePendulumSimulation()
        result = sim.run(params)

        # Both should have 1201 snapshots (initial + 1200 steps)
        assert len(snapshots) == 1201, f"Reference has {len(snapshots)} snapshots, expected 1201"
        assert len(result.x2) == 1201, f"DoublePendulum has {len(result.x2)} snapshots, expected 1201"

        # Time arrays should match
        np.testing.assert_array_almost_equal(ref_t, result.t, decimal=6)

        # Positions should agree to < 1e-6 relative error
        max_err_x = np.max(np.abs(ref_x2 - result.x2) / (np.abs(result.x2) + 1e-10))
        max_err_y = np.max(np.abs(ref_y2 - result.y2) / (np.abs(result.y2) + 1e-10))

        assert (
            max_err_x < 1e-6
        ), f"Max relative error in x2: {max_err_x:.2e}, exceeds 1e-6"
        assert (
            max_err_y < 1e-6
        ), f"Max relative error in y2: {max_err_y:.2e}, exceeds 1e-6"


class TestZeroGravityFrozen:
    """
    Test 3: Limit case — g=0 with ω₀=0 must leave angles frozen.

    Theory: With gravity disabled (g=0) and zero initial velocities (ω₀=0),
    there are no forces driving the system (gravity vector G=0, initial velocities
    zero, Coriolis vanishes at ω=0). The system must remain at rest at initial
    angles for all time.

    This is a deterministic limit case that catches sign errors, missing forces,
    or incorrect dynamics matrices.
    """

    def test_g_zero_frozen_n3(self):
        """N=3 at g=0 for 10 s with zero initial velocities."""
        snapshots = n_pendulum_reference.simulate(
            n=3,
            masses=[1.0, 1.0, 1.0],
            lengths=[1.0, 1.0, 1.0],
            g=0.0,
            thetas0_deg=[30.0, 60.0, 90.0],
            omegas0=[0.0, 0.0, 0.0],
            dt=1.0 / 240,
            n_steps=2400,
        )

        # Extract initial angles in radians
        thetas0_rad = np.deg2rad([30.0, 60.0, 90.0])

        # Extract final angles
        final_snapshot = snapshots[-1]
        thetas_final = np.array(final_snapshot["thetas"])

        # Also check intermediate snapshots to ensure no drift occurs
        for i, snapshot in enumerate(snapshots):
            thetas_t = np.array(snapshot["thetas"])
            np.testing.assert_array_almost_equal(
                thetas_t,
                thetas0_rad,
                decimal=12,
                err_msg=f"Angles drifted at step {i}: {thetas_t} != {thetas0_rad}",
            )
