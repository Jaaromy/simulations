# CSS Tooltip Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the native-browser `<abbr title="...">[?]</abbr>` tooltip pattern across all 7 notebooks with a styled CSS tooltip that appears instantly on hover with no browser delay.

**Architecture:** Each notebook gets one new style-injection cell (a `mo.Html` containing a `<style>` block) inserted near the top. All existing `<abbr title="TEXT">[?]</abbr>` occurrences in `label=` strings are replaced with `<span class="tip" data-tip="TEXT">?</span>`. The CSS lives only in each notebook — no shared file, no extra dependencies.

**Tech Stack:** Marimo 0.23.1, HTML/CSS (`position: absolute`, `::after` pseudo-element, `opacity` transition), Python 3.12.

---

## CSS to use in every notebook

Insert this exact style block (copy verbatim into each task's style cell):

```python
mo.Html("""<style>
.tip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  position: relative;
  cursor: help;
  color: #6b7280;
  font-size: 0.7em;
  border: 1px solid #9ca3af;
  border-radius: 50%;
  width: 1.5em;
  height: 1.5em;
  margin-left: 4px;
  vertical-align: middle;
  font-style: normal;
  user-select: none;
}
.tip::after {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: #1f2937;
  color: #f9fafb;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 1.2rem;
  width: 240px;
  white-space: normal;
  line-height: 1.4;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s ease;
  z-index: 9999;
  text-align: left;
}
.tip:hover::after { opacity: 1; }
</style>""")
```

---

## Label pattern substitution

Old: `label='Foo <abbr title="Some explanation.">[?]</abbr>'`
New: `label='Foo <span class="tip" data-tip="Some explanation.">?</span>'`

---

## Task 1: n_pendulum.py

**Files:**
- Modify: `notebooks/n_pendulum.py`

- [ ] **Step 1: Add style cell**

In `notebooks/n_pendulum.py`, find the imports cell:

```python
@app.cell
def _():
    import marimo as mo
    return (mo,)
```

Add a new cell immediately after it:

```python
@app.cell
def _(mo):
    mo.Html("""<style>
.tip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  position: relative;
  cursor: help;
  color: #6b7280;
  font-size: 0.7em;
  border: 1px solid #9ca3af;
  border-radius: 50%;
  width: 1.5em;
  height: 1.5em;
  margin-left: 4px;
  vertical-align: middle;
  font-style: normal;
  user-select: none;
}
.tip::after {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: #1f2937;
  color: #f9fafb;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 1.2rem;
  width: 240px;
  white-space: normal;
  line-height: 1.4;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s ease;
  z-index: 9999;
  text-align: left;
}
.tip:hover::after { opacity: 1; }
</style>""")
    return
```

- [ ] **Step 2: Replace tooltip markup in labels**

In the controls cell, replace all 6 `<abbr title="...">` with `<span class="tip" data-tip="...">?</span>`:

```python
    n_links_slider = mo.ui.slider(start=1, stop=5, step=1, value=2, label='Links <span class="tip" data-tip="Number of rigid links in the pendulum chain. 1 = simple pendulum (periodic); 2+ = chaotic for large angles.">?</span>')
    g_slider = mo.ui.slider(start=0, stop=20, step=0.1, value=9.81, label='Gravity (m/s²) <span class="tip" data-tip="Gravitational acceleration. 0 = zero-g (pendulum drifts freely); 9.81 = Earth; 20 = strong pull.">?</span>')
    scale_slider = mo.ui.slider(start=50, stop=300, step=10, value=180, label='Scale (px/m) <span class="tip" data-tip="Pixels per metre. Controls how large the pendulum appears on screen. Does not affect the physics.">?</span>')
    trail_toggle = mo.ui.checkbox(value=True, label='Show trails <span class="tip" data-tip="Draw the path traced by the tip of the last link. Useful for visualising chaotic vs periodic motion.">?</span>')
    trail_length_slider = mo.ui.slider(start=50, stop=500, step=50, value=200, label='Trail length <span class="tip" data-tip="Number of recent tip positions to keep in the trail. Longer trails show more history but can obscure the current position.">?</span>')
    ic_textarea = mo.ui.text_area(
        value="120, -60\n120.2, -60\n119.8, -60.1\n",
        label='Initial angles (°) — one pendulum per line, N angles per line <span class="tip" data-tip="Each line defines one pendulum. Provide N comma-separated angles (degrees from downward vertical) for an N-link chain. Multiple nearly-identical lines visualise chaos: tiny differences diverge rapidly.">?</span>',
    )
```

- [ ] **Step 3: Verify notebook starts without errors**

```bash
uv run marimo run notebooks/n_pendulum.py
```

Expected: notebook opens in browser, tooltips appear on hover over `?` badges, no console errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/n_pendulum.py
git commit -m "feat: CSS tooltips in n_pendulum notebook"
```

---

## Task 2: double_pendulum.py

**Files:**
- Modify: `notebooks/double_pendulum.py`

- [ ] **Step 1: Add style cell**

Find the `import marimo as mo` cell and add the style cell immediately after it (same CSS block as Task 1, Step 1 — copy verbatim).

- [ ] **Step 2: Replace tooltip markup in labels**

Find the controls cell and replace all 4 `<abbr title="...">` occurrences:

```python
    theta1_slider = mo.ui.slider(start=-180, stop=180, step=1, value=120, label='θ₁ — upper bob angle (°) <span class="tip" data-tip="Angle of the upper rod from the downward vertical. 0° = hanging straight down. 180° = pointing straight up.">?</span>')
    theta2_slider = mo.ui.slider(start=-180, stop=180, step=1, value=-60, label='θ₂ — lower bob angle (°) <span class="tip" data-tip="Angle of the lower rod from the downward vertical. Setting this close to θ₁ gives regular motion; large differences produce chaos.">?</span>')
    duration_slider = mo.ui.slider(start=5, stop=60, step=5, value=20, label='Duration (s) <span class="tip" data-tip="How long to simulate. Longer runs reveal more of the chaotic trajectory.">?</span>')
    chaos_toggle = mo.ui.checkbox(value=False, label='Show chaos comparison <span class="tip" data-tip="Runs a second pendulum with θ₁ perturbed by 0.01°. Both trajectories start indistinguishable but diverge exponentially — visualising sensitive dependence on initial conditions.">?</span>')
```

(Read the file first to confirm exact variable names and slider bounds before editing.)

- [ ] **Step 3: Verify notebook starts without errors**

```bash
uv run marimo run notebooks/double_pendulum.py
```

Expected: notebook opens, tooltips work, no errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/double_pendulum.py
git commit -m "feat: CSS tooltips in double_pendulum notebook"
```

---

## Task 3: ball_collisions.py

**Files:**
- Modify: `notebooks/ball_collisions.py`

- [ ] **Step 1: Add style cell**

Find the `import marimo as mo` cell and add the style cell immediately after it (same CSS block as Task 1, Step 1 — copy verbatim).

- [ ] **Step 2: Replace tooltip markup in labels**

Read the file first (`Read notebooks/ball_collisions.py`) to get exact current label strings, then replace all 8 `<abbr title="...">` occurrences with `<span class="tip" data-tip="...">?</span>`. The tooltip text stays identical — only the HTML wrapper changes.

Pattern to apply to each label:
- Remove: `<abbr title="TOOLTIP_TEXT">[?]</abbr>`
- Insert: `<span class="tip" data-tip="TOOLTIP_TEXT">?</span>`

- [ ] **Step 3: Verify notebook starts without errors**

```bash
uv run marimo run notebooks/ball_collisions.py
```

Expected: notebook opens, tooltips work, no errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/ball_collisions.py
git commit -m "feat: CSS tooltips in ball_collisions notebook"
```

---

## Task 4: casino_comparison.py

**Files:**
- Modify: `notebooks/casino_comparison.py`

- [ ] **Step 1: Add style cell**

Find the `import marimo as mo` cell and add the style cell immediately after it (same CSS block as Task 1, Step 1 — copy verbatim).

- [ ] **Step 2: Replace tooltip markup in labels**

Read the file first (`Read notebooks/casino_comparison.py`) to get exact current label strings, then replace all 7 `<abbr title="...">` occurrences with `<span class="tip" data-tip="...">?</span>`. The tooltip text stays identical — only the HTML wrapper changes.

- [ ] **Step 3: Verify notebook starts without errors**

```bash
uv run marimo run notebooks/casino_comparison.py
```

Expected: notebook opens, tooltips work, no errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/casino_comparison.py
git commit -m "feat: CSS tooltips in casino_comparison notebook"
```

---

## Task 5: logistic_population_growth.py

**Files:**
- Modify: `notebooks/logistic_population_growth.py`

- [ ] **Step 1: Add style cell**

Find the `import marimo as mo` cell and add the style cell immediately after it (same CSS block as Task 1, Step 1 — copy verbatim).

- [ ] **Step 2: Replace tooltip markup in labels**

Read the file first (`Read notebooks/logistic_population_growth.py`) to get exact current label strings, then replace all 6 `<abbr title="...">` occurrences with `<span class="tip" data-tip="...">?</span>`. The tooltip text stays identical — only the HTML wrapper changes.

- [ ] **Step 3: Verify notebook starts without errors**

```bash
uv run marimo run notebooks/logistic_population_growth.py
```

Expected: notebook opens, tooltips work, no errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/logistic_population_growth.py
git commit -m "feat: CSS tooltips in logistic_population_growth notebook"
```

---

## Task 6: ising_exploration.py

**Files:**
- Modify: `notebooks/ising_exploration.py`

- [ ] **Step 1: Add style cell**

Find the `import marimo as mo` cell and add the style cell immediately after it (same CSS block as Task 1, Step 1 — copy verbatim).

- [ ] **Step 2: Replace tooltip markup in labels**

Read the file first (`Read notebooks/ising_exploration.py`) to get exact current label strings, then replace all 4 `<abbr title="...">` occurrences with `<span class="tip" data-tip="...">?</span>`. The tooltip text stays identical — only the HTML wrapper changes.

- [ ] **Step 3: Verify notebook starts without errors**

```bash
uv run marimo run notebooks/ising_exploration.py
```

Expected: notebook opens, tooltips work, no errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/ising_exploration.py
git commit -m "feat: CSS tooltips in ising_exploration notebook"
```

---

## Task 7: sir_epidemic.py

**Files:**
- Modify: `notebooks/sir_epidemic.py`

- [ ] **Step 1: Add style cell**

Find the `import marimo as mo` cell and add the style cell immediately after it (same CSS block as Task 1, Step 1 — copy verbatim).

- [ ] **Step 2: Replace tooltip markup in labels**

Read the file first (`Read notebooks/sir_epidemic.py`) to get exact current label strings, then replace all 5 `<abbr title="...">` occurrences with `<span class="tip" data-tip="...">?</span>`. The tooltip text stays identical — only the HTML wrapper changes.

- [ ] **Step 3: Verify notebook starts without errors**

```bash
uv run marimo run notebooks/sir_epidemic.py
```

Expected: notebook opens, tooltips work, no errors.

- [ ] **Step 4: Commit**

```bash
git add notebooks/sir_epidemic.py
git commit -m "feat: CSS tooltips in sir_epidemic notebook"
```

---

## Task 8: Final verification and PR

- [ ] **Step 1: Confirm no `<abbr` remains**

```bash
grep -r "<abbr" notebooks/
```

Expected: no output (zero matches).

- [ ] **Step 2: Confirm all notebooks have style cell**

```bash
grep -l "class=\"tip\"" notebooks/*.py
```

Expected: all 7 notebook files listed.

- [ ] **Step 3: Push branch and open PR**

```bash
git push -u origin feat/css-tooltips
gh pr create --title "Replace abbr tooltips with CSS tooltips across all notebooks" --body "$(cat <<'EOF'
## Summary
- Adds a CSS `.tip` style cell to all 7 notebooks
- Replaces native `<abbr title>` tooltips (delayed, unstyled) with instant CSS tooltips
- Tooltip text is unchanged — only the HTML wrapper and delivery mechanism change

## Test plan
- [ ] Open each notebook with `uv run marimo run notebooks/<name>.py`
- [ ] Hover over `?` badge next to each parameter — tooltip card appears instantly
- [ ] Confirm no `<abbr` tags remain: `grep -r "<abbr" notebooks/`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
