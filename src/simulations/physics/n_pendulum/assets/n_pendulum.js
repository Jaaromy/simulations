/**
 * n_pendulum.js — N-link Lagrangian pendulum physics engine and canvas renderer.
 *
 * Physics model (Lagrangian, N links):
 *   State: [theta_0, omega_0, theta_1, omega_1, ..., theta_{N-1}, omega_{N-1}]
 *   Angles measured from downward vertical, positive counterclockwise.
 *
 *   Mass matrix M (N×N):
 *     M[i][j] = tailMass(max(i,j)) * L[i] * L[j] * cos(theta[i] - theta[j])
 *     tailMass(k) = m[k] + m[k+1] + ... + m[N-1]
 *
 *   Coriolis/centripetal C (N-vector):
 *     C[i] = sum_j tailMass(max(i,j)) * L[i] * L[j] * sin(theta[i] - theta[j]) * omega[j]^2
 *
 *   Gravity G (N-vector):
 *     G[i] = tailMass(i) * g * L[i] * sin(theta[i])
 *
 *   Solve: M * alpha = -(C + G)  for angular accelerations alpha.
 *
 *   Integrator: 4th-order Runge-Kutta (RK4) with fixed dt.
 *
 * Exported to window scope (browser) and module.exports (Node.js).
 *
 * No external dependencies — pure vanilla JS.
 */

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

/**
 * Gaussian elimination (no pivoting). Solves Ax = b in-place.
 * Returns solution vector x, or null if singular.
 * For N<=5 the mass matrix is positive definite for physical configs,
 * so no pivoting is needed.
 */
function gaussianElimination(A, b) {
  var n = b.length;
  // Augmented matrix [A | b]
  var M = [];
  for (var i = 0; i < n; i++) {
    M[i] = A[i].slice();
    M[i].push(b[i]);
  }

  // Forward elimination
  for (var col = 0; col < n; col++) {
    var pivot = M[col][col];
    if (Math.abs(pivot) < 1e-15) {
      // Try to find a non-zero row below and swap
      var swapped = false;
      for (var row = col + 1; row < n; row++) {
        if (Math.abs(M[row][col]) > 1e-15) {
          var tmp = M[col]; M[col] = M[row]; M[row] = tmp;
          pivot = M[col][col];
          swapped = true;
          break;
        }
      }
      if (!swapped) return null; // singular
    }
    for (var row = col + 1; row < n; row++) {
      var factor = M[row][col] / pivot;
      for (var k = col; k <= n; k++) {
        M[row][k] -= factor * M[col][k];
      }
    }
  }

  // Back substitution
  var x = new Array(n);
  for (var i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (var j = i + 1; j < n; j++) {
      x[i] -= M[i][j] * x[j];
    }
    x[i] /= M[i][i];
  }
  return x;
}

/**
 * Scalar multiply array by scalar s.
 */
function scaleVec(v, s) {
  var out = new Array(v.length);
  for (var i = 0; i < v.length; i++) out[i] = v[i] * s;
  return out;
}

/**
 * Add two arrays element-wise.
 */
function addVec(a, b) {
  var out = new Array(a.length);
  for (var i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

// ---------------------------------------------------------------------------
// Physics core
// ---------------------------------------------------------------------------

/**
 * Build tail-mass lookup: tailMass[k] = sum_{j=k}^{N-1} masses[j].
 */
function buildTailMasses(masses) {
  var N = masses.length;
  var tm = new Array(N + 1);
  tm[N] = 0.0;
  for (var k = N - 1; k >= 0; k--) {
    tm[k] = tm[k + 1] + masses[k];
  }
  return tm;
}

/**
 * Extract thetas and omegas from flat state vector.
 * State layout: [theta_0, omega_0, theta_1, omega_1, ...]
 */
function extractState(state) {
  var N = state.length / 2;
  var thetas = new Array(N);
  var omegas = new Array(N);
  for (var i = 0; i < N; i++) {
    thetas[i] = state[2 * i];
    omegas[i] = state[2 * i + 1];
  }
  return { thetas: thetas, omegas: omegas };
}

/**
 * Compute derivatives of state for RK4.
 * Returns flat array [omega_0, alpha_0, omega_1, alpha_1, ...]
 */
function derivs(state, masses, lengths, g, tailMasses) {
  var N = masses.length;
  var s = extractState(state);
  var thetas = s.thetas;
  var omegas = s.omegas;

  // Build mass matrix M (N×N)
  var Mmat = [];
  for (var i = 0; i < N; i++) {
    Mmat[i] = new Array(N);
    for (var j = 0; j < N; j++) {
      var k = Math.max(i, j);
      Mmat[i][j] = tailMasses[k] * lengths[i] * lengths[j] * Math.cos(thetas[i] - thetas[j]);
    }
  }

  // Build Coriolis/centripetal vector C
  var C = new Array(N).fill(0);
  for (var i = 0; i < N; i++) {
    for (var j = 0; j < N; j++) {
      var k = Math.max(i, j);
      C[i] += tailMasses[k] * lengths[i] * lengths[j] * Math.sin(thetas[i] - thetas[j]) * omegas[j] * omegas[j];
    }
  }

  // Build gravity vector G
  var G = new Array(N);
  for (var i = 0; i < N; i++) {
    G[i] = tailMasses[i] * g * lengths[i] * Math.sin(thetas[i]);
  }

  // RHS = -(C + G)
  var rhs = new Array(N);
  for (var i = 0; i < N; i++) {
    rhs[i] = -(C[i] + G[i]);
  }

  // Solve M * alpha = rhs
  var alpha = gaussianElimination(Mmat, rhs);
  if (!alpha) {
    // Fallback: zero accelerations (degenerate config)
    alpha = new Array(N).fill(0);
  }

  // Pack into flat derivative vector
  var dstate = new Array(state.length);
  for (var i = 0; i < N; i++) {
    dstate[2 * i] = omegas[i];      // d(theta_i)/dt = omega_i
    dstate[2 * i + 1] = alpha[i];   // d(omega_i)/dt = alpha_i
  }
  return dstate;
}

/**
 * Single RK4 step.
 */
function rk4Step(state, masses, lengths, g, tailMasses, dt) {
  var k1 = derivs(state, masses, lengths, g, tailMasses);
  var k2 = derivs(addVec(state, scaleVec(k1, dt / 2)), masses, lengths, g, tailMasses);
  var k3 = derivs(addVec(state, scaleVec(k2, dt / 2)), masses, lengths, g, tailMasses);
  var k4 = derivs(addVec(state, scaleVec(k3, dt)), masses, lengths, g, tailMasses);

  var newState = new Array(state.length);
  for (var i = 0; i < state.length; i++) {
    newState[i] = state[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }
  return newState;
}

/**
 * Compute Cartesian positions of each bob.
 * Returns arrays x, y in metres (from pivot at origin, y down positive).
 *   x_k = sum_{j=0}^{k} L[j] * sin(theta[j])
 *   y_k = sum_{j=0}^{k} L[j] * cos(theta[j])   (positive = downward)
 * Note: canvas y increases downward; caller handles coordinate flip for display.
 */
function cartesianPositions(thetas, lengths) {
  var N = thetas.length;
  var x = new Array(N);
  var y = new Array(N);
  var sx = 0, sy = 0;
  for (var k = 0; k < N; k++) {
    sx += lengths[k] * Math.sin(thetas[k]);
    sy += lengths[k] * Math.cos(thetas[k]);
    x[k] = sx;
    y[k] = sy;
  }
  return { x: x, y: y };
}

/**
 * Compute total mechanical energy.
 *   T = 0.5 * sum_i sum_j tailMass(max(i,j)) * L[i] * L[j] * omega[i] * omega[j] * cos(theta[i]-theta[j])
 *   V = -sum_i tailMass(i) * g * L[i] * cos(theta[i])
 *   E = T + V
 */
function computeEnergy(thetas, omegas, masses, lengths, g, tailMasses) {
  var N = masses.length;
  var T = 0;
  for (var i = 0; i < N; i++) {
    for (var j = 0; j < N; j++) {
      var k = Math.max(i, j);
      T += tailMasses[k] * lengths[i] * lengths[j] * omegas[i] * omegas[j] * Math.cos(thetas[i] - thetas[j]);
    }
  }
  T *= 0.5;

  var V = 0;
  for (var i = 0; i < N; i++) {
    V -= tailMasses[i] * g * lengths[i] * Math.cos(thetas[i]);
  }

  return T + V;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

var BACKGROUND_COLOR = '#0d0d1a';
var PIVOT_RADIUS = 5;
var BOB_RADIUS = 6;
var OUTERMOST_BOB_RADIUS = 8;
var ARM_LINE_WIDTH = 2.5;
var TRAIL_LINE_WIDTH = 1.5;
var TRAIL_MAX_ALPHA = 0.6;

/**
 * Generate distinct colours for each pendulum using evenly spaced HSL hues
 * starting at 200 degrees.
 */
function pendulumColor(pendulumIndex, totalPendulums, lightness) {
  var hue = (200 + pendulumIndex * (360 / Math.max(totalPendulums, 1))) % 360;
  return 'hsl(' + hue + ', 100%, ' + lightness + '%)';
}

/**
 * Draw a single frame on the canvas.
 */
function drawFrame(ctx, canvas, pendulumStates, config, trails) {
  var N = config.n;
  var cx = canvas.width / 2;
  var cy = canvas.height / 2;
  var scale = config.scalePxPerM;
  var numPendulums = pendulumStates.length;

  // Clear canvas
  ctx.fillStyle = BACKGROUND_COLOR;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw trails
  if (config.trailEnabled) {
    for (var p = 0; p < numPendulums; p++) {
      var trail = trails[p];
      var trailLen = trail.length;
      if (trailLen < 2) continue;

      var hue = (200 + p * (360 / Math.max(numPendulums, 1))) % 360;
      ctx.lineWidth = TRAIL_LINE_WIDTH;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      for (var t = 1; t < trailLen; t++) {
        // Fade from 0 (oldest) to TRAIL_MAX_ALPHA (newest)
        var alpha = (t / (trailLen - 1)) * TRAIL_MAX_ALPHA;
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = 'hsl(' + hue + ', 100%, 60%)';
        ctx.beginPath();
        ctx.moveTo(trail[t - 1][0], trail[t - 1][1]);
        ctx.lineTo(trail[t][0], trail[t][1]);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1.0;
  }

  // Draw arms and bobs for each pendulum
  for (var p = 0; p < numPendulums; p++) {
    var state = pendulumStates[p];
    var s = extractState(state);
    var thetas = s.thetas;
    var pos = cartesianPositions(thetas, config.lengths);

    var hue = (200 + p * (360 / Math.max(numPendulums, 1))) % 360;
    var armColor = 'hsl(' + hue + ', 80%, 70%)';
    var bobColor = 'hsl(' + hue + ', 100%, 75%)';

    // Draw arms
    ctx.strokeStyle = armColor;
    ctx.lineWidth = ARM_LINE_WIDTH;
    ctx.globalAlpha = 1.0;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    for (var k = 0; k < N; k++) {
      ctx.lineTo(cx + pos.x[k] * scale, cy - pos.y[k] * scale);
    }
    ctx.stroke();

    // Draw bobs
    for (var k = 0; k < N; k++) {
      var bx = cx + pos.x[k] * scale;
      var by = cy - pos.y[k] * scale;
      var radius = (k === N - 1) ? OUTERMOST_BOB_RADIUS : BOB_RADIUS;
      ctx.fillStyle = bobColor;
      ctx.beginPath();
      ctx.arc(bx, by, radius, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // Draw pivot
  ctx.fillStyle = '#ffffff';
  ctx.globalAlpha = 1.0;
  ctx.beginPath();
  ctx.arc(cx, cy, PIVOT_RADIUS, 0, 2 * Math.PI);
  ctx.fill();
}

// ---------------------------------------------------------------------------
// Ring buffer for trails
// ---------------------------------------------------------------------------

/**
 * A ring buffer storing the last `capacity` positions.
 * Positions are [x, y] canvas coordinates.
 */
function createRingBuffer(capacity) {
  return {
    capacity: capacity,
    data: [],
    start: 0,
    count: 0
  };
}

function ringBufferPush(rb, item) {
  if (rb.count < rb.capacity) {
    rb.data.push(item);
    rb.count++;
  } else {
    rb.data[rb.start] = item;
    rb.start = (rb.start + 1) % rb.capacity;
  }
}

/**
 * Get all items in order (oldest to newest).
 */
function ringBufferToArray(rb) {
  if (rb.count < rb.capacity) {
    return rb.data.slice();
  }
  var result = new Array(rb.capacity);
  for (var i = 0; i < rb.capacity; i++) {
    result[i] = rb.data[(rb.start + i) % rb.capacity];
  }
  return result;
}

// ---------------------------------------------------------------------------
// Public API: start()
// ---------------------------------------------------------------------------

/**
 * Start the animation loop.
 *
 * config: {
 *   canvasId: string,
 *   n: number,             // number of links (1-5)
 *   masses: number[],
 *   lengths: number[],
 *   g: number,
 *   pendulums: [{thetas: number[], omegas: number[]}],  // degrees, rad/s
 *   dt: number,
 *   substeps: number,
 *   trailEnabled: boolean,
 *   trailLength: number,
 *   scalePxPerM: number,
 * }
 *
 * Returns { getTrajectory(pendulumIndex) } for testing.
 */
function start(config) {
  var canvas = document.getElementById(config.canvasId);
  var ctx = canvas.getContext('2d');
  var N = config.n;
  var tailMasses = buildTailMasses(config.masses);
  var numPendulums = config.pendulums.length;

  // Convert initial conditions: degrees to radians
  var states = config.pendulums.map(function(p) {
    var state = new Array(2 * N);
    for (var i = 0; i < N; i++) {
      state[2 * i] = (p.thetas[i] * Math.PI) / 180;
      state[2 * i + 1] = p.omegas[i];
    }
    return state;
  });

  // Trail ring buffers
  var trails = states.map(function() {
    return createRingBuffer(config.trailLength || 200);
  });

  // Trajectory storage for testing (stores all positions)
  var trajectories = states.map(function() { return []; });

  var animId = null;

  function frame() {
    // Physics substeps
    for (var step = 0; step < config.substeps; step++) {
      for (var p = 0; p < numPendulums; p++) {
        states[p] = rk4Step(states[p], config.masses, config.lengths, config.g, tailMasses, config.dt);
      }
    }

    // Append outermost bob position to trails
    var cx = canvas.width / 2;
    var cy = canvas.height / 2;
    for (var p = 0; p < numPendulums; p++) {
      var s = extractState(states[p]);
      var pos = cartesianPositions(s.thetas, config.lengths);
      var outerX = cx + pos.x[N - 1] * config.scalePxPerM;
      var outerY = cy - pos.y[N - 1] * config.scalePxPerM;
      ringBufferPush(trails[p], [outerX, outerY]);

      // Store for getTrajectory
      trajectories[p].push({ thetas: s.thetas.slice(), omegas: s.omegas.slice(), x: pos.x.slice(), y: pos.y.slice() });
    }

    var trailArrays = trails.map(function(rb) { return ringBufferToArray(rb); });
    drawFrame(ctx, canvas, states, config, trailArrays);

    animId = requestAnimationFrame(frame);
  }

  animId = requestAnimationFrame(frame);

  return {
    getTrajectory: function(pendulumIndex) {
      return trajectories[pendulumIndex] || [];
    },
    stop: function() {
      if (animId !== null) {
        cancelAnimationFrame(animId);
        animId = null;
      }
    }
  };
}

// ---------------------------------------------------------------------------
// Headless simulate() for testing (no canvas)
// ---------------------------------------------------------------------------

/**
 * Run simulation headlessly and return snapshots.
 *
 * config: {
 *   n: number,
 *   masses: number[],
 *   lengths: number[],
 *   g: number,
 *   thetas_deg: number[],  // initial angles in degrees
 *   omegas: number[],      // initial angular velocities in rad/s
 *   dt: number,
 *   n_steps: number,
 * }
 *
 * Returns array of n_steps+1 snapshots:
 * [{ t, thetas, omegas, x, y, energy }, ...]
 *   x, y: arrays of N cartesian positions in metres (not scaled).
 *         x positive right, y positive downward from pivot.
 */
function simulate(config) {
  var N = config.n;
  var masses = config.masses;
  var lengths = config.lengths;
  var g = config.g;
  var dt = config.dt;
  var nSteps = config.n_steps;
  var tailMasses = buildTailMasses(masses);

  // Initial state (degrees to radians)
  var state = new Array(2 * N);
  for (var i = 0; i < N; i++) {
    state[2 * i] = (config.thetas_deg[i] * Math.PI) / 180;
    state[2 * i + 1] = config.omegas[i];
  }

  var snapshots = new Array(nSteps + 1);

  function snapshot(t, st) {
    var s = extractState(st);
    var pos = cartesianPositions(s.thetas, lengths);
    var energy = computeEnergy(s.thetas, s.omegas, masses, lengths, g, tailMasses);
    return {
      t: t,
      thetas: s.thetas.slice(),
      omegas: s.omegas.slice(),
      x: pos.x.slice(),
      y: pos.y.slice(),
      energy: energy
    };
  }

  snapshots[0] = snapshot(0, state);

  for (var step = 0; step < nSteps; step++) {
    state = rk4Step(state, masses, lengths, g, tailMasses, dt);
    snapshots[step + 1] = snapshot((step + 1) * dt, state);
  }

  return snapshots;
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

// Browser: attach to window
if (typeof window !== 'undefined') {
  window.NPendulum = { start: start };
}

// Node.js: CommonJS export for tests
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { simulate: simulate };
}
