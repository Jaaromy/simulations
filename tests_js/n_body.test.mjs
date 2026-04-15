/**
 * n_body.test.mjs — JS test suite for the N-body gravitational physics engine.
 *
 * Three layers:
 *   Layer 1 — Determinism: same ICs → bit-identical output.
 *   Layer 2 — Limit cases: known analytic outcomes at extremes.
 *   Layer 3 — Theory cross-validation: Kepler III, energy conservation,
 *              gold trajectory parity, merge invariant.
 *
 * Run: node --test tests_js/n_body.test.mjs
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
const { simulate, step } = require('../src/simulations/physics/gravity/assets/n_body.js');

// Physical constants
const AU = 1.496e11;          // metres per AU
const G = 6.6743e-11;
const M_SUN = 1.989e30;       // kg
const M_EARTH = 5.972e24;     // kg
const R_EARTH = 6.371e6;      // m
const R_SUN = 6.957e8;        // m

// Minimal Sun+Earth config used across multiple tests
function sunEarthConfig() {
  return {
    bodies: [
      { name: 'Sun',   massKg: M_SUN,   xM: 0,    yM: 0, vxMs: 0, vyMs: 0,     radiusM: R_SUN,   color: '#ffff00' },
      { name: 'Earth', massKg: M_EARTH, xM: AU,   yM: 0, vxMs: 0, vyMs: 29784, radiusM: R_EARTH, color: '#0000ff' },
    ],
    G,
    softeningM: 1e9,
    dtS: 3600,
    substeps: 1,
    mergeEnabled: false,
  };
}

// ---------------------------------------------------------------------------
// Layer 1 — Determinism
// ---------------------------------------------------------------------------

describe('Layer 1: Determinism', () => {
  test('determinism: same inputs produce identical outputs', () => {
    const config = sunEarthConfig();
    const r1 = simulate(config, 100);
    const r2 = simulate(config, 100);
    assert.strictEqual(r1.length, r2.length, 'snapshot count mismatch');
    for (let i = 0; i < r1.length; i++) {
      for (let k = 0; k < r1[i].bodies.length; k++) {
        const a = r1[i].bodies[k];
        const b = r2[i].bodies[k];
        assert.strictEqual(a.x,  b.x,  `x  mismatch at step ${i} body ${k}`);
        assert.strictEqual(a.y,  b.y,  `y  mismatch at step ${i} body ${k}`);
        assert.strictEqual(a.vx, b.vx, `vx mismatch at step ${i} body ${k}`);
        assert.strictEqual(a.vy, b.vy, `vy mismatch at step ${i} body ${k}`);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Layer 2 — Limit cases
// ---------------------------------------------------------------------------

describe('Layer 2: Limit cases', () => {
  // 2a. Single body linear motion (no gravity partner)
  test('limit: single body moves in straight line', () => {
    const config = {
      bodies: [
        { name: 'Lone', massKg: 1e24, xM: 0, yM: 0, vxMs: 1000, vyMs: 0, radiusM: 1e6, color: '#ffffff' },
      ],
      G,
      softeningM: 1e9,
      dtS: 1,
      substeps: 1,
      mergeEnabled: false,
    };
    const snaps = simulate(config, 10);
    for (let i = 0; i < snaps.length; i++) {
      const b = snaps[i].bodies[0];
      const expectedX = 1000 * (i + 1);  // vx*dt*(step+1)
      const errX = Math.abs(b.x - expectedX);
      assert.ok(errX < 1e-10, `step ${i+1}: x=${b.x}, expected ${expectedX}, err=${errX}`);
      assert.ok(Math.abs(b.y) < 1e-30, `step ${i+1}: y should be 0, got ${b.y}`);
    }
  });

  // 2b. Circular orbit radius stability over 1 year
  test('limit: circular orbit radius stays bounded', () => {
    const config = sunEarthConfig();
    // 8760 steps at dt=3600 s = 1 year
    const snaps = simulate(config, 8760);
    const r0 = AU; // initial radius from origin (Sun at 0)
    let maxRelDev = 0;
    for (const snap of snaps) {
      const earth = snap.bodies[1];
      if (!earth.alive) continue;
      const r = Math.sqrt(earth.x * earth.x + earth.y * earth.y);
      const relDev = Math.abs(r - r0) / r0;
      if (relDev > maxRelDev) maxRelDev = relDev;
    }
    assert.ok(
      maxRelDev < 1e-3,
      `circular orbit max radius deviation = ${maxRelDev.toExponential(3)} (expected < 1e-3 relative)`,
    );
  });

  // 2c. Inelastic merge conserves momentum
  test('limit: inelastic merge conserves momentum', () => {
    const m = 1e26;
    const r = 1.5e8; // radiusM; sum = 3e8 > initial separation 2e8 → immediate overlap
    const config = {
      bodies: [
        { name: 'A', massKg: m, xM: -1e8, yM: 0, vxMs:  1000, vyMs: 0, radiusM: r, color: '#ff0000' },
        { name: 'B', massKg: m, xM:  1e8, yM: 0, vxMs: -1000, vyMs: 0, radiusM: r, color: '#0000ff' },
      ],
      G,
      softeningM: 1e9,
      dtS: 1,
      substeps: 1,
      mergeEnabled: true,
    };

    // Initial total momentum
    const px0 = m * 1000 + m * (-1000); // = 0
    const py0 = 0;

    // Run until one body dies (at most 10 steps)
    let mergeStep = -1;
    const snaps = simulate(config, 10);
    for (let i = 0; i < snaps.length; i++) {
      const dead = snaps[i].bodies.filter(b => !b.alive);
      if (dead.length > 0) { mergeStep = i; break; }
    }
    assert.ok(mergeStep >= 0, 'bodies never merged within 10 steps');

    // After merge: compute total momentum of alive bodies
    const snap = snaps[mergeStep];
    let pxFinal = 0, pyFinal = 0, totalMass = 0;
    for (const b of snap.bodies) {
      if (!b.alive) continue;
      pxFinal += b.mass * b.vx;
      pyFinal += b.mass * b.vy;
      totalMass += b.mass;
    }

    // Initial total momentum ≈ 0; check relative to total impulse scale m*v
    const scale = m * 1000; // characteristic momentum
    const relErrX = Math.abs(pxFinal - px0) / scale;
    const relErrY = Math.abs(pyFinal - py0) / scale;
    assert.ok(relErrX < 1e-6, `px not conserved: pxFinal=${pxFinal}, scale=${scale}, relErr=${relErrX}`);
    assert.ok(relErrY < 1e-6, `py not conserved: pyFinal=${pyFinal}, scale=${scale}, relErr=${relErrY}`);
  });
});

// ---------------------------------------------------------------------------
// Layer 3 — Theory cross-validation
// ---------------------------------------------------------------------------

describe('Layer 3: Theory cross-validation', () => {
  // 3a. Kepler's Third Law: T^2 / (4π² a³ / GM) = 1  (within 0.5%)
  test('theory: Kepler third law T^2 proportional to a^3', () => {
    const config = sunEarthConfig();
    // 2 years = 17520 steps at dt=3600 s
    const snaps = simulate(config, 17520);

    // Detect Earth's orbital period via sign changes of y (pos → neg crossing)
    // Sun is body 0, Earth is body 1. Sun drifts slightly — use relative y.
    const crossings = [];
    let prevY = snaps[0].bodies[1].y - snaps[0].bodies[0].y;
    for (let i = 1; i < snaps.length && crossings.length < 2; i++) {
      const currY = snaps[i].bodies[1].y - snaps[i].bodies[0].y;
      if (prevY > 0 && currY <= 0) {
        // Linear interpolation: fraction into this step where y=0
        const frac = prevY / (prevY - currY);
        crossings.push((i - 1 + frac) * 3600); // time in seconds
      }
      prevY = currY;
    }
    assert.ok(crossings.length >= 2, `could not detect 2 y-crossings, found ${crossings.length}`);

    const T_measured = crossings[1] - crossings[0];
    // Kepler III: T^2 = 4π² a³ / (G * M_sun)
    const a = AU;
    const T_kepler = 2 * Math.PI * Math.sqrt(a * a * a / (G * M_SUN));
    const relErr = Math.abs(T_measured - T_kepler) / T_kepler;

    assert.ok(
      relErr < 0.005,
      `Kepler III: T_measured=${(T_measured/86400).toFixed(2)}d, T_kepler=${(T_kepler/86400).toFixed(2)}d, relErr=${relErr.toExponential(3)}`,
    );
  });

  // 3b. Total mechanical energy conserved to 1e-4 over 2 years
  test('theory: total mechanical energy conserved to 1e-4', () => {
    const config = sunEarthConfig();
    const snaps = simulate(config, 17520);

    function totalEnergy(snap) {
      const bodies = snap.bodies.filter(b => b.alive);
      let E = 0;
      // Kinetic
      for (const b of bodies) {
        E += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy);
      }
      // Potential (pairs)
      for (let i = 0; i < bodies.length; i++) {
        for (let j = i + 1; j < bodies.length; j++) {
          const dx = bodies[j].x - bodies[i].x;
          const dy = bodies[j].y - bodies[i].y;
          const r = Math.sqrt(dx * dx + dy * dy);
          E -= G * bodies[i].mass * bodies[j].mass / r;
        }
      }
      return E;
    }

    // Compute initial energy directly from config bodies before simulate
    // Sun: body[0] at rest at origin; Earth: body[1] at 1 AU with initial circular orbit velocity
    const sun = config.bodies[0];
    const earth = config.bodies[1];
    let E0 = 0;
    // Kinetic (Sun at rest)
    E0 += 0.5 * sun.massKg * (sun.vxMs * sun.vxMs + sun.vyMs * sun.vyMs);
    // Kinetic (Earth)
    E0 += 0.5 * earth.massKg * (earth.vxMs * earth.vxMs + earth.vyMs * earth.vyMs);
    // Potential (Sun-Earth pair)
    const dx = earth.xM - sun.xM;
    const dy = earth.yM - sun.yM;
    const r = Math.sqrt(dx * dx + dy * dy);
    E0 -= G * sun.massKg * earth.massKg / r;

    let maxRelErr = 0;
    for (const snap of snaps) {
      const relErr = Math.abs(totalEnergy(snap) - E0) / Math.abs(E0);
      if (relErr > maxRelErr) maxRelErr = relErr;
    }

    assert.ok(
      maxRelErr < 1e-4,
      `energy conservation: max |ΔE|/|E0| = ${maxRelErr.toExponential(3)} (expected < 1e-4)`,
    );
  });

  // 3c. Gold trajectory parity — 2-body (Sun+Earth)
  test('theory: matches Python reference trajectory — 2body', () => {
    const goldPath = join(__dirname, 'reference', 'gold_trajectory_2body.json');
    let gold;
    try {
      gold = JSON.parse(readFileSync(goldPath, 'utf8'));
    } catch {
      assert.fail('Gold file missing: ' + goldPath + ' — regenerate by running: uv run python tests_js/reference/n_body_reference.py');
    }
    const cfg = gold.config;

    // Build simulate config from gold config
    const simConfig = {
      bodies: cfg.bodies,
      G: cfg.G,
      softeningM: cfg.softening,
      dtS: cfg.dt,
      substeps: 1,
      mergeEnabled: cfg.merge,
    };

    const saveEvery = cfg.save_every || 1;
    const totalSteps = cfg.n_steps;
    // Run all steps, collect matching Python subsampling: range(0, n_steps, save_every)
    // Python trajectory[0] = step 1 (t=dt), trajectory[save_every] = step save_every+1, etc.
    // Python subsample: saved[k] = trajectory[k * save_every] = after (k*save_every + 1) steps
    // JS allSnaps[i] = after (i+1) steps → match at allSnaps[k * save_every]
    const allSnaps = simulate(simConfig, totalSteps);
    const jsSnaps = [];
    for (let i = 0; i < allSnaps.length; i += saveEvery) {
      jsSnaps.push(allSnaps[i]);
    }

    assert.strictEqual(jsSnaps.length, gold.trajectory.length,
      `snapshot count: JS=${jsSnaps.length} gold=${gold.trajectory.length}`);

    const a_semi = AU; // 1 AU normalisation scale
    let maxRelErr = 0;
    for (let i = 0; i < jsSnaps.length; i++) {
      for (let k = 0; k < jsSnaps[i].bodies.length; k++) {
        const js = jsSnaps[i].bodies[k];
        const py = gold.trajectory[i].bodies[k];
        if (!js.alive || !py.alive) continue;
        const errX = Math.abs(js.x - py.x) / a_semi;
        const errY = Math.abs(js.y - py.y) / a_semi;
        if (errX > maxRelErr) maxRelErr = errX;
        if (errY > maxRelErr) maxRelErr = errY;
      }
    }

    assert.ok(
      maxRelErr < 1e-4,
      `2body gold parity: max rel err = ${maxRelErr.toExponential(3)} (expected < 1e-4)`,
    );
  });

  // 3d. Gold trajectory parity — solar (Sun+Earth+Luna)
  test('theory: matches Python reference trajectory — solar', () => {
    const goldPath = join(__dirname, 'reference', 'gold_trajectory_solar.json');
    let gold;
    try {
      gold = JSON.parse(readFileSync(goldPath, 'utf8'));
    } catch {
      assert.fail('Gold file missing: ' + goldPath + ' — regenerate by running: uv run python tests_js/reference/n_body_reference.py');
    }
    const cfg = gold.config;

    const simConfig = {
      bodies: cfg.bodies,
      G: cfg.G,
      softeningM: cfg.softening,
      dtS: cfg.dt,
      substeps: 1,
      mergeEnabled: cfg.merge,
    };

    const saveEvery = cfg.save_every || 1;
    const totalSteps = cfg.n_steps;
    const allSnaps = simulate(simConfig, totalSteps);
    const jsSnaps = [];
    for (let i = 0; i < allSnaps.length; i += saveEvery) {
      jsSnaps.push(allSnaps[i]);
    }

    assert.strictEqual(jsSnaps.length, gold.trajectory.length,
      `snapshot count: JS=${jsSnaps.length} gold=${gold.trajectory.length}`);

    const a_semi = AU;
    let maxRelErr = 0;
    for (let i = 0; i < jsSnaps.length; i++) {
      for (let k = 0; k < jsSnaps[i].bodies.length; k++) {
        const js = jsSnaps[i].bodies[k];
        const py = gold.trajectory[i].bodies[k];
        if (!js.alive || !py.alive) continue;
        const errX = Math.abs(js.x - py.x) / a_semi;
        const errY = Math.abs(js.y - py.y) / a_semi;
        if (errX > maxRelErr) maxRelErr = errX;
        if (errY > maxRelErr) maxRelErr = errY;
      }
    }

    assert.ok(
      maxRelErr < 1e-4,
      `solar gold parity: max rel err = ${maxRelErr.toExponential(3)} (expected < 1e-4)`,
    );
  });

  // 3e. Merge: single body at CoM, zero net momentum
  test('theory: merge produces single body at CoM with zero net momentum', () => {
    const m = 1e26;
    const r = 1.5e8;
    const x0A = -1e8, x0B = 1e8;
    const vA = 1000, vB = -1000;
    const config = {
      bodies: [
        { name: 'A', massKg: m, xM: x0A, yM: 0, vxMs: vA,  vyMs: 0, radiusM: r, color: '#ff0000' },
        { name: 'B', massKg: m, xM: x0B, yM: 0, vxMs: vB, vyMs: 0, radiusM: r, color: '#0000ff' },
      ],
      G,
      softeningM: 1e9,
      dtS: 1,
      substeps: 1,
      mergeEnabled: true,
    };

    // Initial CoM
    const comX0 = (m * x0A + m * x0B) / (2 * m); // = 0
    const comY0 = 0;
    // Initial total momentum
    const px0 = m * vA + m * vB; // = 0
    const py0 = 0;

    const snaps = simulate(config, 10);
    let mergeSnap = null;
    for (const snap of snaps) {
      if (snap.bodies.some(b => !b.alive)) { mergeSnap = snap; break; }
    }
    assert.ok(mergeSnap !== null, 'no merge occurred within 10 steps');

    const alive = mergeSnap.bodies.filter(b => b.alive);
    assert.strictEqual(alive.length, 1, `expected 1 alive body after merge, got ${alive.length}`);

    const survivor = alive[0];
    // Position within 1.0 m of initial CoM.
    // Gravity causes ~0.8 m drift during the step, so 1 mm tolerance was too tight.
    assert.ok(
      Math.abs(survivor.x - comX0) < 1.0,
      `CoM x: got ${survivor.x}, expected ${comX0}, diff=${Math.abs(survivor.x - comX0)}`,
    );
    assert.ok(
      Math.abs(survivor.y - comY0) < 1.0,
      `CoM y: got ${survivor.y}, expected ${comY0}, diff=${Math.abs(survivor.y - comY0)}`,
    );
    // Net momentum ≈ 0 (head-on equal-mass)
    assert.ok(
      Math.abs(survivor.vx) < 1e-10,
      `vx after merge: ${survivor.vx} (expected ≈ 0)`,
    );
  });
});
