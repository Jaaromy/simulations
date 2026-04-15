/**
 * n_pendulum.test.mjs — JS test suite for the N-link pendulum physics engine.
 *
 * Three layers:
 *   Layer 1 — Determinism: same ICs → bit-identical output.
 *   Layer 2 — Limit cases: known analytic outcomes at extremes.
 *   Layer 3 — Theory cross-validation: energy conservation + gold trajectory parity.
 *
 * Run: node --test tests_js/n_pendulum.test.mjs
 *
 * Sign convention note:
 *   JS engine: y[k] = +sum L[j]*cos(theta[j])  (y positive downward)
 *   Gold JSON: y[k] = -sum L[j]*cos(theta[j])  (y positive upward)
 *   → negate JS y when comparing against gold.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Load the JS engine (CommonJS module)
const { simulate } = require('../src/simulations/physics/n_pendulum/assets/n_pendulum.js');

// ---------------------------------------------------------------------------
// Layer 1 — Determinism
// ---------------------------------------------------------------------------

describe('Layer 1: Determinism', () => {
  test('same ICs produce bit-identical trajectories', () => {
    const config = {
      n: 2, masses: [1, 1], lengths: [1, 1], g: 9.81,
      thetas_deg: [120, -60], omegas: [0, 0], dt: 1 / 240, n_steps: 100,
    };
    const r1 = simulate(config);
    const r2 = simulate(config);
    assert.strictEqual(r1.length, r2.length, 'snapshot count mismatch');
    for (let i = 0; i < r1.length; i++) {
      for (let k = 0; k < 2; k++) {
        assert.strictEqual(r1[i].x[k], r2[i].x[k], `x mismatch at step ${i} bob ${k}`);
        assert.strictEqual(r1[i].y[k], r2[i].y[k], `y mismatch at step ${i} bob ${k}`);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Layer 2 — Limit cases
// ---------------------------------------------------------------------------

describe('Layer 2: Limit cases', () => {
  // 2a. g=0, all ω=0 → angles frozen forever
  test('2a: g=0 with zero omegas keeps angles frozen (N=3, 10s)', () => {
    const thetas_deg = [30, 60, 90];
    const config = {
      n: 3, masses: [1, 1, 1], lengths: [0.8, 0.7, 0.6], g: 0,
      thetas_deg, omegas: [0, 0, 0], dt: 1 / 240, n_steps: 2400,
    };
    const result = simulate(config);
    const initial = result[0].thetas;
    const final = result[result.length - 1].thetas;
    for (let k = 0; k < 3; k++) {
      const diff = Math.abs(final[k] - initial[k]);
      assert.ok(diff < 1e-12, `theta[${k}] drifted by ${diff} (expected < 1e-12)`);
    }
  });

  // 2b. Simple pendulum period (N=1, small angle ≈5°)
  test('2b: N=1 small-angle period matches theoretical 2π√(L/g) within 0.5%', () => {
    const g = 9.81;
    const L = 1.0;
    const T_theory = 2 * Math.PI * Math.sqrt(L / g); // ≈ 2.006 s

    // Run for 2 full periods to find second positive crossing
    const n_steps = Math.round(2 * T_theory * 240) + 1;
    const config = {
      n: 1, masses: [1.0], lengths: [L], g,
      thetas_deg: [5.0], omegas: [0.0], dt: 1 / 240, n_steps,
    };
    const result = simulate(config);

    // Detect period: find two consecutive negative→positive zero-crossings of theta[0].
    // First crossing is at t ≈ T/2 (positive→negative→positive).
    // Second crossing is at t ≈ 3T/2. Difference = T.
    // Alternatively: track time between first and second neg→pos crossings directly.
    let negPosCrossings = [];
    let prevTheta = result[0].thetas[0];
    for (let i = 1; i < result.length && negPosCrossings.length < 2; i++) {
      const curr = result[i].thetas[0];
      if (prevTheta < 0 && curr >= 0) {
        // Linear interpolation for sub-step precision
        const frac = -prevTheta / (curr - prevTheta);
        negPosCrossings.push(result[i - 1].t + frac * (1 / 240));
      }
      prevTheta = curr;
    }

    assert.ok(negPosCrossings.length >= 2, 'Could not detect two period crossings');
    const T_detected = negPosCrossings[1] - negPosCrossings[0];
    const relErr = Math.abs(T_detected - T_theory) / T_theory;
    assert.ok(relErr < 0.005, `Detected period ${T_detected.toFixed(4)}s, theory ${T_theory.toFixed(4)}s, relErr=${relErr.toFixed(5)} (expected < 0.5%)`);
  });

  // 2c. Near-degenerate N=2 (m2→0) behaves like simple pendulum
  test('2c: N=2 near-degenerate (m2=0.001, L2=0.001) period within 1% of simple pendulum', () => {
    const g = 9.81;
    const L = 1.0;
    const T_theory = 2 * Math.PI * Math.sqrt(L / g); // ≈ 2.006 s

    const n_steps = Math.round(2 * T_theory * 240) + 1;
    const config = {
      n: 2, masses: [1.0, 0.001], lengths: [1.0, 0.001], g,
      thetas_deg: [5.0, 0.0], omegas: [0, 0], dt: 1 / 240, n_steps,
    };
    const result = simulate(config);

    let negPosCrossings = [];
    let prevTheta = result[0].thetas[0];
    for (let i = 1; i < result.length && negPosCrossings.length < 2; i++) {
      const curr = result[i].thetas[0];
      if (prevTheta < 0 && curr >= 0) {
        const frac = -prevTheta / (curr - prevTheta);
        negPosCrossings.push(result[i - 1].t + frac * (1 / 240));
      }
      prevTheta = curr;
    }

    assert.ok(negPosCrossings.length >= 2, 'Could not detect two period crossings');
    const T_detected = negPosCrossings[1] - negPosCrossings[0];
    const relErr = Math.abs(T_detected - T_theory) / T_theory;
    assert.ok(relErr < 0.01, `Detected period ${T_detected.toFixed(4)}s, theory ${T_theory.toFixed(4)}s, relErr=${relErr.toFixed(5)} (expected < 1%)`);
  });
});

// ---------------------------------------------------------------------------
// Layer 3 — Theory cross-validation
// ---------------------------------------------------------------------------

describe('Layer 3: Theory cross-validation', () => {
  // 3a. Energy conservation for N ∈ {2, 3, 4, 5}
  describe('3a: Energy conservation over 10s (max |ΔE|/|E0| < 0.1%)', () => {
    const configs = [
      {
        label: 'N=2',
        n: 2, masses: [1, 1], lengths: [1, 1],
        thetas_deg: [120, -60], omegas: [0, 0],
      },
      {
        label: 'N=3',
        n: 3, masses: [1, 1, 1], lengths: [0.8, 0.7, 0.6],
        thetas_deg: [120, -60, 45], omegas: [0, 0, 0],
      },
      {
        label: 'N=4',
        n: 4, masses: [1, 1, 1, 1], lengths: [0.7, 0.6, 0.5, 0.4],
        thetas_deg: [120, -60, 45, -30], omegas: [0, 0, 0, 0],
      },
      // N=5 has more degrees of freedom; RK4 at dt=1/240 yields slightly higher drift.
      // Threshold is loosened to 0.2% (still a tight physics anchor).
      {
        label: 'N=5',
        n: 5, masses: [1, 1, 1, 1, 1], lengths: [0.6, 0.5, 0.4, 0.35, 0.3],
        thetas_deg: [120, -60, 45, -30, 20], omegas: [0, 0, 0, 0, 0],
        threshold: 0.002,
      },
    ];

    for (const cfg of configs) {
      test(`energy conservation ${cfg.label}`, () => {
        const result = simulate({
          ...cfg,
          g: 9.81, dt: 1 / 240, n_steps: 2400,
        });
        const threshold = cfg.threshold ?? 0.001;
        const E0 = result[0].energy;
        let maxRelErr = 0;
        for (const snap of result) {
          const relErr = Math.abs(snap.energy - E0) / Math.abs(E0);
          if (relErr > maxRelErr) maxRelErr = relErr;
        }
        assert.ok(
          maxRelErr < threshold,
          `${cfg.label}: max |ΔE|/|E0| = ${maxRelErr.toExponential(3)} (expected < ${threshold})`,
        );
      });
    }
  });

  // 3b. N=2 parity vs Python gold trajectory
  test('3b: N=2 matches gold trajectory (outermost bob, max rel err < 1e-4)', () => {
    const goldPath = join(__dirname, 'reference', 'gold_trajectory_n2.json');
    const gold = JSON.parse(readFileSync(goldPath, 'utf8'));

    const result = simulate({
      n: 2, masses: [1, 1], lengths: [1, 1], g: 9.81,
      thetas_deg: [120, -60], omegas: [0, 0], dt: 1 / 240, n_steps: 1200,
    });

    assert.strictEqual(result.length, gold.length, 'snapshot count mismatch');

    // JS y is +sum L*cos(theta) (positive downward).
    // Gold y is -sum L*cos(theta) (positive upward).
    // Negate JS y before comparison.
    let maxRelErr = 0;
    for (let i = 0; i < result.length; i++) {
      const jsX = result[i].x[1];
      const jsY = -result[i].y[1]; // flip sign to match gold convention
      const goldX = gold[i].x[1];
      const goldY = gold[i].y[1];

      const errX = Math.abs(jsX - goldX) / (Math.abs(goldX) + 1e-9);
      const errY = Math.abs(jsY - goldY) / (Math.abs(goldY) + 1e-9);
      if (errX > maxRelErr) maxRelErr = errX;
      if (errY > maxRelErr) maxRelErr = errY;
    }

    assert.ok(
      maxRelErr < 1e-4,
      `N=2 gold parity: max rel err = ${maxRelErr.toExponential(3)} (expected < 1e-4)`,
    );
  });

  // 3c. N=3 parity vs Python gold trajectory
  test('3c: N=3 matches gold trajectory (outermost bob, max rel err < 1e-4)', () => {
    const goldPath = join(__dirname, 'reference', 'gold_trajectory_n3.json');
    const gold = JSON.parse(readFileSync(goldPath, 'utf8'));

    const result = simulate({
      n: 3, masses: [1, 1, 1], lengths: [0.8, 0.7, 0.6], g: 9.81,
      thetas_deg: [120, -60, 45], omegas: [0, 0, 0], dt: 1 / 240, n_steps: 1200,
    });

    assert.strictEqual(result.length, gold.length, 'snapshot count mismatch');

    // Same sign flip as 3b.
    let maxRelErr = 0;
    for (let i = 0; i < result.length; i++) {
      const jsX = result[i].x[2];
      const jsY = -result[i].y[2];
      const goldX = gold[i].x[2];
      const goldY = gold[i].y[2];

      const errX = Math.abs(jsX - goldX) / (Math.abs(goldX) + 1e-9);
      const errY = Math.abs(jsY - goldY) / (Math.abs(goldY) + 1e-9);
      if (errX > maxRelErr) maxRelErr = errX;
      if (errY > maxRelErr) maxRelErr = errY;
    }

    assert.ok(
      maxRelErr < 1e-4,
      `N=3 gold parity: max rel err = ${maxRelErr.toExponential(3)} (expected < 1e-4)`,
    );
  });
});
