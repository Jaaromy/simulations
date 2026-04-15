// N-body gravitational simulation — real-time browser renderer + headless Node export
// Physics: Velocity-Verlet integrator (symplectic, energy-conserving)

const AU = 1.496e11; // metres per astronomical unit
const SUN_RADIUS_M = 6.957e8;
const EARTH_MASS_KG = 5.972e24;
const EARTH_RADIUS_M = 6.371e6;
const BASE_PX = 8; // display radius base for log scale
const REF_RADIUS_M = SUN_RADIUS_M;

// ─── Physics ──────────────────────────────────────────────────────────────────

function computeAccelerations(bodies, G, softeningM) {
  const n = bodies.length;
  const ax = new Float64Array(n);
  const ay = new Float64Array(n);
  const eps2 = softeningM * softeningM;
  for (let i = 0; i < n; i++) {
    if (!bodies[i].alive) continue;
    for (let j = 0; j < n; j++) {
      if (j === i || !bodies[j].alive) continue;
      const dx = bodies[j].x - bodies[i].x;
      const dy = bodies[j].y - bodies[i].y;
      const dist2 = dx * dx + dy * dy + eps2;
      const dist3 = dist2 * Math.sqrt(dist2);
      const f = G * bodies[j].mass / dist3;
      ax[i] += f * dx;
      ay[i] += f * dy;
    }
  }
  return { ax, ay };
}

// Single Velocity-Verlet step (modifies bodies array in-place, returns it)
function step(state, config) {
  const { G, softeningM, dtS, mergeEnabled } = config;
  const dt = dtS;

  // Deep-copy bodies so step is pure
  const bodies = state.bodies.map(b => Object.assign({}, b));

  // Step 1: accelerations at current positions
  const { ax: ax0, ay: ay0 } = computeAccelerations(bodies, G, softeningM);

  // Step 2: update positions
  for (let i = 0; i < bodies.length; i++) {
    if (!bodies[i].alive) continue;
    bodies[i].x += bodies[i].vx * dt + 0.5 * ax0[i] * dt * dt;
    bodies[i].y += bodies[i].vy * dt + 0.5 * ay0[i] * dt * dt;
  }

  // Step 3: recompute accelerations at new positions
  const { ax: ax1, ay: ay1 } = computeAccelerations(bodies, G, softeningM);

  // Step 4: update velocities
  for (let i = 0; i < bodies.length; i++) {
    if (!bodies[i].alive) continue;
    bodies[i].vx += 0.5 * (ax0[i] + ax1[i]) * dt;
    bodies[i].vy += 0.5 * (ay0[i] + ay1[i]) * dt;
  }

  // Step 5: merge collisions
  if (mergeEnabled) {
    for (let i = 0; i < bodies.length; i++) {
      if (!bodies[i].alive) continue;
      for (let j = i + 1; j < bodies.length; j++) {
        if (!bodies[j].alive) continue;
        const dx = bodies[j].x - bodies[i].x;
        const dy = bodies[j].y - bodies[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < bodies[i].radius + bodies[j].radius) {
          const mi = bodies[i].mass;
          const mj = bodies[j].mass;
          const mt = mi + mj;
          // conserve momentum
          bodies[i].vx = (mi * bodies[i].vx + mj * bodies[j].vx) / mt;
          bodies[i].vy = (mi * bodies[i].vy + mj * bodies[j].vy) / mt;
          // center of mass position
          bodies[i].x = (mi * bodies[i].x + mj * bodies[j].x) / mt;
          bodies[i].y = (mi * bodies[i].y + mj * bodies[j].y) / mt;
          // conserve volume → radius
          bodies[i].radius = Math.cbrt(
            bodies[i].radius ** 3 + bodies[j].radius ** 3
          );
          bodies[i].mass = mt;
          // keep name/color of larger body
          if (mj > mi) {
            bodies[i].name = bodies[j].name;
            bodies[i].color = bodies[j].color;
          }
          bodies[j].alive = false;
        }
      }
    }
  }

  return { bodies };
}

// Pure headless simulation — returns array of snapshots
function simulate(config, nSteps) {
  const { bodies: rawBodies } = config;
  const bodies = rawBodies.map(b => ({
    name: b.name,
    mass: b.massKg,
    x: b.xM,
    y: b.yM,
    vx: b.vxMs,
    vy: b.vyMs,
    radius: b.radiusM,
    color: b.color || '#ffffff',
    alive: true,
  }));

  let state = { bodies };
  const snapshots = [];

  for (let s = 0; s < nSteps; s++) {
    state = step(state, config);
    snapshots.push({
      bodies: state.bodies.map(b => ({
        x: b.x,
        y: b.y,
        vx: b.vx,
        vy: b.vy,
        mass: b.mass,
        alive: b.alive,
      })),
    });
  }

  return snapshots;
}

// Returns the number of substeps needed so each substep is ≤ 5% of the
// dynamical timescale t_dyn = sqrt(r³ / G(mᵢ+mⱼ)) for the closest live pair.
function requiredSubsteps(bodies, G, dtS, softeningM, baseSubsteps) {
  let maxRatio = 1;
  for (let i = 0; i < bodies.length; i++) {
    if (!bodies[i].alive) continue;
    for (let j = i + 1; j < bodies.length; j++) {
      if (!bodies[j].alive) continue;
      const dx = bodies[j].x - bodies[i].x;
      const dy = bodies[j].y - bodies[i].y;
      const r2 = dx * dx + dy * dy + softeningM * softeningM;
      const r = Math.sqrt(r2);
      const tDyn = Math.sqrt(r * r2 / (G * (bodies[i].mass + bodies[j].mass)));
      const ratio = dtS / tDyn;
      if (ratio > maxRatio) maxRatio = ratio;
    }
  }
  // Each substep should be ≤ 5% of t_dyn
  return Math.min(Math.max(baseSubsteps, Math.ceil(maxRatio / 0.05)), 100000);
}

function computeTotalEnergy(bodies, G, softeningM) {
  const eps2 = softeningM * softeningM;
  let KE = 0, PE = 0;
  for (let i = 0; i < bodies.length; i++) {
    if (!bodies[i].alive) continue;
    KE += 0.5 * bodies[i].mass * (bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy);
    for (let j = i + 1; j < bodies.length; j++) {
      if (!bodies[j].alive) continue;
      const dx = bodies[j].x - bodies[i].x;
      const dy = bodies[j].y - bodies[i].y;
      const dist = Math.sqrt(dx * dx + dy * dy + eps2);
      PE -= G * bodies[i].mass * bodies[j].mass / dist;
    }
  }
  return KE + PE;
}

// ─── Browser renderer ─────────────────────────────────────────────────────────

(function () {
  if (typeof window === 'undefined') return; // Node — skip browser code

  let animId = null;
  let originalConfig = null;
  let paused = false;

  // Trail ring buffer per body
  function makeTrail(length) {
    return { buf: [], maxLen: length };
  }
  function pushTrail(trail, x, y) {
    trail.buf.push({ x, y });
    if (trail.buf.length > trail.maxLen) trail.buf.shift();
  }

  function displayRadius(radiusM, logRadiusScale, metersPerPixel) {
    if (logRadiusScale) {
      return Math.max(2, BASE_PX * Math.log(1 + radiusM / REF_RADIUS_M));
    }
    return Math.max(2, radiusM / metersPerPixel);
  }

  function getCameraPos(bodies, viewCenter) {
    if (viewCenter === 'sun') {
      const sun = bodies.find(b => b.alive);
      return sun ? { cx: sun.x, cy: sun.y } : { cx: 0, cy: 0 };
    }
    // mass-weighted centroid
    let totalMass = 0, cx = 0, cy = 0;
    for (const b of bodies) {
      if (!b.alive) continue;
      cx += b.mass * b.x;
      cy += b.mass * b.y;
      totalMass += b.mass;
    }
    return totalMass > 0
      ? { cx: cx / totalMass, cy: cy / totalMass }
      : { cx: 0, cy: 0 };
  }

  function worldToCanvas(wx, wy, camX, camY, metersPerPixel, canvasWidth, canvasHeight) {
    return {
      px: canvasWidth / 2 + (wx - camX) / metersPerPixel,
      py: canvasHeight / 2 - (wy - camY) / metersPerPixel,
    };
  }

  function start(config) {
    if (animId !== null) cancelAnimationFrame(animId);
    originalConfig = config;
    paused = false;

    const canvas = document.getElementById(config.canvasId);
    if (!canvas) { console.error('NBody: canvas not found:', config.canvasId); return; }
    const ctx = canvas.getContext('2d');
    canvas.width = config.canvasWidth;
    canvas.height = config.canvasHeight;

    // Build mutable body state
    const bodies = config.bodies.map(b => ({
      name: b.name,
      mass: b.massKg,
      x: b.xM,
      y: b.yM,
      vx: b.vxMs,
      vy: b.vyMs,
      radius: b.radiusM,
      color: b.color || '#ffffff',
      alive: true,
      trail: makeTrail(config.trailLength || 200),
    }));

    const metersPerPixel = (config.viewScaleAu * AU * 2) / config.canvasWidth;
    const substeps = config.substeps || 1;
    const timeWarp = config.timeWarp || 1;

    let initialEnergy = null;
    let currentEnergyDrift = 0;
    let currentSubsteps = substeps;

    // ── Controls UI ──
    injectControls(canvas, config, bodies);

    // ── Animation loop ──
    let lastTimestamp = null;

    function frame(timestamp) {
      animId = requestAnimationFrame(frame);
      if (paused) { lastTimestamp = null; return; }

      // Real seconds elapsed since last frame (cap at 0.1s to avoid spiral-of-death)
      const frameDt = lastTimestamp !== null
        ? Math.min((timestamp - lastTimestamp) / 1000, 0.1)
        : 1 / 60;
      lastTimestamp = timestamp;

      if (initialEnergy === null) {
        initialEnergy = computeTotalEnergy(bodies, config.G, config.softeningM);
      }

      // Advance simulation with adaptive per-step timestepping.
      // Recompute substeps each integration step so close approaches developing
      // mid-frame are caught before energy error accumulates.
      const simSecondsNeeded = timeWarp * frameDt;
      const MAX_STEPS_PER_FRAME = 5000; // prevents browser stall
      let simTimeElapsed = 0;
      let stepsDone = 0;

      while (simTimeElapsed < simSecondsNeeded - 1e-9 && stepsDone < MAX_STEPS_PER_FRAME) {
        currentSubsteps = requiredSubsteps(bodies, config.G, config.dtS, config.softeningM, substeps);
        const dtStep = Math.min(
          config.dtS / currentSubsteps,
          simSecondsNeeded - simTimeElapsed,
        );
        const aliveBefore = bodies.reduce((n, b) => n + (b.alive ? 1 : 0), 0);
        const stepCfg = Object.assign({}, config, { dtS: dtStep });
        const newState = step({ bodies }, stepCfg);
        for (let i = 0; i < bodies.length; i++) {
          Object.assign(bodies[i], newState.bodies[i]);
        }
        // Inelastic merge intentionally loses KE — reset energy baseline so drift
        // meter tracks only numerical error from the post-merge state onward.
        const aliveAfter = bodies.reduce((n, b) => n + (b.alive ? 1 : 0), 0);
        if (aliveAfter !== aliveBefore) initialEnergy = null;
        simTimeElapsed += dtStep;
        stepsDone++;
      }

      if (initialEnergy !== null && initialEnergy !== 0) {
        const E = computeTotalEnergy(bodies, config.G, config.softeningM);
        currentEnergyDrift = (E - initialEnergy) / Math.abs(initialEnergy) * 100;
      }

      // Update trails
      if (config.trailEnabled) {
        for (const b of bodies) {
          if (b.alive) pushTrail(b.trail, b.x, b.y);
        }
      }

      // Render
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#000010';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const { cx: camX, cy: camY } = getCameraPos(bodies, config.viewCenter);

      for (const b of bodies) {
        if (!b.alive) continue;
        const { px, py } = worldToCanvas(b.x, b.y, camX, camY, metersPerPixel, canvas.width, canvas.height);
        const dispR = displayRadius(b.radius, config.logRadiusScale, metersPerPixel);

        // Trail
        if (config.trailEnabled && b.trail.buf.length > 1) {
          ctx.beginPath();
          ctx.strokeStyle = b.color + '66'; // ~40% opacity
          ctx.lineWidth = 1;
          const first = b.trail.buf[0];
          const fp = worldToCanvas(first.x, first.y, camX, camY, metersPerPixel, canvas.width, canvas.height);
          ctx.moveTo(fp.px, fp.py);
          for (let k = 1; k < b.trail.buf.length; k++) {
            const tp = worldToCanvas(b.trail.buf[k].x, b.trail.buf[k].y, camX, camY, metersPerPixel, canvas.width, canvas.height);
            ctx.lineTo(tp.px, tp.py);
          }
          ctx.stroke();
        }

        // Body disc
        ctx.beginPath();
        ctx.arc(px, py, dispR, 0, Math.PI * 2);
        ctx.fillStyle = b.color;
        ctx.fill();

        // Label
        if (config.showLabels) {
          ctx.fillStyle = '#ffffff';
          ctx.font = '11px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(b.name, px, py - dispR - 3);
        }
      }

      // Accuracy-limited warning (step budget exhausted this frame)
      if (stepsDone >= MAX_STEPS_PER_FRAME) {
        ctx.font = '11px monospace';
        ctx.textAlign = 'left';
        ctx.fillStyle = '#000010cc';
        const warnText = '⚠ slowing for accuracy';
        const ww = ctx.measureText(warnText).width;
        ctx.fillRect(8, 6, ww + 10, 18);
        ctx.fillStyle = '#ffcc00';
        ctx.fillText(warnText, 13, 19);
      }

      // Energy drift + substep overlay
      if (initialEnergy !== null) {
        const absDrift = Math.abs(currentEnergyDrift);
        const driftColor = absDrift < 0.1 ? '#44ff88' : absDrift < 1.0 ? '#ffcc00' : '#ff4444';
        const driftLabel = absDrift < 0.1
          ? `ΔE/E₀: ${currentEnergyDrift.toFixed(3)}%`
          : absDrift < 1.0
          ? `ΔE/E₀: ${currentEnergyDrift.toFixed(2)}% ⚠`
          : `ΔE/E₀: ${currentEnergyDrift.toFixed(1)}% ✕`;
        const subLabel = `sub: ${currentSubsteps}`;
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        // drift line
        ctx.fillStyle = '#000010cc';
        const tw1 = ctx.measureText(driftLabel).width;
        ctx.fillRect(canvas.width - tw1 - 14, 6, tw1 + 10, 18);
        ctx.fillStyle = driftColor;
        ctx.fillText(driftLabel, canvas.width - 8, 19);
        // substep line
        const tw2 = ctx.measureText(subLabel).width;
        ctx.fillStyle = '#000010cc';
        ctx.fillRect(canvas.width - tw2 - 14, 26, tw2 + 10, 18);
        ctx.fillStyle = currentSubsteps > substeps ? '#ffcc00' : '#888888';
        ctx.fillText(subLabel, canvas.width - 8, 39);
      }

      // Draw drag arrow if active
      if (dragState.active) {
        const sp = worldToCanvas(dragState.wx, dragState.wy, camX, camY, metersPerPixel, canvas.width, canvas.height);
        // Preview circle showing body's rendered size
        const _massInput = document.getElementById('nbody-mass-input');
        const _massEarths = _massInput ? parseFloat(_massInput.value) || 1.0 : 1.0;
        const previewR = displayRadius(EARTH_RADIUS_M * Math.cbrt(_massEarths), config.logRadiusScale, metersPerPixel);
        ctx.beginPath();
        ctx.arc(sp.px, sp.py, previewR, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
        // Velocity arrow
        ctx.beginPath();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.moveTo(sp.px, sp.py);
        ctx.lineTo(dragState.ex, dragState.ey);
        ctx.stroke();
        // arrowhead
        const angle = Math.atan2(dragState.ey - sp.py, dragState.ex - sp.px);
        const headLen = 10;
        ctx.beginPath();
        ctx.moveTo(dragState.ex, dragState.ey);
        ctx.lineTo(dragState.ex - headLen * Math.cos(angle - 0.4), dragState.ey - headLen * Math.sin(angle - 0.4));
        ctx.lineTo(dragState.ex - headLen * Math.cos(angle + 0.4), dragState.ey - headLen * Math.sin(angle + 0.4));
        ctx.closePath();
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    }

    // ── Click-drag to add body ──
    const VELOCITY_SCALE = 30000 / 100; // 300 m/s per pixel of drag → 100 px ≈ Earth orbital speed
    const dragState = { active: false, wx: 0, wy: 0, ex: 0, ey: 0 };

    function canvasToWorld(px, py, camX, camY) {
      const wx = camX + (px - canvas.width / 2) * metersPerPixel;
      const wy = camY - (py - canvas.height / 2) * metersPerPixel;
      return { wx, wy };
    }

    function getCanvasPos(e) {
      const rect = canvas.getBoundingClientRect();
      return { cpx: e.clientX - rect.left, cpy: e.clientY - rect.top };
    }

    canvas.addEventListener('mousedown', e => {
      const { cpx, cpy } = getCanvasPos(e);
      const { cx: camX, cy: camY } = getCameraPos(bodies, config.viewCenter);
      const { wx, wy } = canvasToWorld(cpx, cpy, camX, camY);
      dragState.active = true;
      dragState.wx = wx;
      dragState.wy = wy;
      dragState.ex = cpx;
      dragState.ey = cpy;
    });

    canvas.addEventListener('mousemove', e => {
      if (!dragState.active) return;
      const { cpx, cpy } = getCanvasPos(e);
      dragState.ex = cpx;
      dragState.ey = cpy;
    });

    canvas.addEventListener('mouseup', e => {
      if (!dragState.active) return;
      dragState.active = false;
      const { cpx, cpy } = getCanvasPos(e);
      const { cx: camX, cy: camY } = getCameraPos(bodies, config.viewCenter);
      const startWorld = { wx: dragState.wx, wy: dragState.wy };
      const endWorld = canvasToWorld(cpx, cpy, camX, camY);
      const dvx = (endWorld.wx - startWorld.wx) * VELOCITY_SCALE / metersPerPixel;
      const dvy = (endWorld.wy - startWorld.wy) * VELOCITY_SCALE / metersPerPixel;

      const massInput = document.getElementById('nbody-mass-input');
      const massEarths = massInput ? parseFloat(massInput.value) || 1.0 : 1.0;
      bodies.push({
        name: 'New',
        mass: massEarths * EARTH_MASS_KG,
        x: startWorld.wx,
        y: startWorld.wy,
        vx: dvx,
        vy: dvy,
        radius: EARTH_RADIUS_M * Math.cbrt(massEarths),
        color: '#ffffff',
        alive: true,
        trail: makeTrail(config.trailLength || 200),
      });
    });

    animId = requestAnimationFrame(frame);
  }

  function injectControls(canvas, config, bodies) {
    const container = canvas.parentElement || document.body;

    // Mass input
    if (!document.getElementById('nbody-mass-input')) {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.id = 'nbody-mass-input';
      inp.value = '1.0';
      inp.step = '0.1';
      inp.min = '0.001';
      inp.title = 'New body mass (Earth masses)';
      inp.style.cssText = 'position:absolute;top:8px;left:8px;width:120px;background:#1a1a2e;color:#fff;border:1px solid #444;padding:4px 6px;border-radius:4px;font-size:12px;z-index:10;';
      const label = document.createElement('label');
      label.style.cssText = 'position:absolute;top:34px;left:8px;color:#aaa;font-size:11px;z-index:10;';
      label.textContent = 'Earth masses (click-drag to add)';
      container.style.position = 'relative';
      container.appendChild(inp);
      container.appendChild(label);
    }

    // Pause/Play button
    if (!document.getElementById('nbody-pause-btn')) {
      const btn = document.createElement('button');
      btn.id = 'nbody-pause-btn';
      btn.textContent = 'Pause';
      btn.style.cssText = 'position:absolute;bottom:8px;left:8px;background:#1a1a2e;color:#fff;border:1px solid #444;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px;z-index:10;';
      btn.addEventListener('click', () => {
        paused = !paused;
        btn.textContent = paused ? 'Play' : 'Pause';
      });
      container.appendChild(btn);
    }

    // Reset button
    if (!document.getElementById('nbody-reset-btn')) {
      const btn = document.createElement('button');
      btn.id = 'nbody-reset-btn';
      btn.textContent = 'Reset';
      btn.style.cssText = 'position:absolute;bottom:8px;left:80px;background:#1a1a2e;color:#fff;border:1px solid #444;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px;z-index:10;';
      btn.addEventListener('click', () => {
        const pauseBtn = document.getElementById('nbody-pause-btn');
        if (pauseBtn) pauseBtn.textContent = 'Pause';
        window.NBody.start(originalConfig);
      });
      container.appendChild(btn);
    }
  }

  window.NBody = { start };
})();

// ─── CommonJS export (Node.js tests) ──────────────────────────────────────────
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { simulate, step };
}
