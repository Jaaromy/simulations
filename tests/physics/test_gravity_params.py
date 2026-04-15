"""
Tests for simulations.physics.gravity.params.

Covers:
  - solar_system_preset() shape and values
  - GravityParams.validate() error cases
  - parse_ic_text / bodies_to_text round-trip
  - parse_ic_text edge cases (comments, blank lines, short lines)
  - GravityParams.to_dict() structure
"""
from __future__ import annotations

import math

import pytest

from simulations.physics.gravity.params import (
    BodyIC,
    GravityParams,
    bodies_to_text,
    parse_ic_text,
    solar_system_preset,
)


# ---------------------------------------------------------------------------
# Preset shape
# ---------------------------------------------------------------------------

class TestSolarSystemPreset:
    """solar_system_preset() must return well-formed data."""

    def test_returns_10_bodies(self):
        bodies = solar_system_preset()
        assert len(bodies) == 10, f"Expected 10 bodies, got {len(bodies)}"

    def test_first_body_is_sun(self):
        bodies = solar_system_preset()
        sun = bodies[0]
        assert sun.name == "Sun"
        assert math.isclose(sun.mass_kg, 1.989e30, rel_tol=0.01), (
            f"Sun mass {sun.mass_kg:.3e} not within 1% of 1.989e30"
        )

    def test_all_bodies_positive_mass_and_radius(self):
        bodies = solar_system_preset()
        for b in bodies:
            assert b.mass_kg > 0, f"{b.name}: mass_kg={b.mass_kg} not positive"
            assert b.radius_m > 0, f"{b.name}: radius_m={b.radius_m} not positive"

    def test_luna_vy_greater_than_earth_vy(self):
        """Luna has Earth's orbital speed plus its own orbital speed around Earth."""
        bodies = solar_system_preset()
        names = [b.name for b in bodies]
        earth = bodies[names.index("Earth")]
        luna = bodies[names.index("Luna")]
        assert luna.vy_m_s > earth.vy_m_s, (
            f"Luna vy_m_s={luna.vy_m_s:.2f} <= Earth vy_m_s={earth.vy_m_s:.2f}"
        )


# ---------------------------------------------------------------------------
# validate() errors
# ---------------------------------------------------------------------------

class TestValidateErrors:
    """GravityParams.validate() must raise ValueError for invalid fields."""

    def test_empty_bodies_raises(self):
        p = GravityParams(bodies=[])
        with pytest.raises(ValueError, match="bodies"):
            p.validate()

    def test_zero_G_raises(self):
        p = GravityParams(G=0)
        with pytest.raises(ValueError, match="G"):
            p.validate()

    def test_negative_dt_raises(self):
        p = GravityParams(dt_s=-1)
        with pytest.raises(ValueError, match="dt_s"):
            p.validate()

    def test_invalid_view_center_raises(self):
        p = GravityParams(view_center="moon")
        with pytest.raises(ValueError, match="view_center"):
            p.validate()

    def test_zero_canvas_width_raises(self):
        p = GravityParams(canvas_width=0)
        with pytest.raises(ValueError, match="canvas"):
            p.validate()


# ---------------------------------------------------------------------------
# parse_ic_text round-trip
# ---------------------------------------------------------------------------

class TestParseRoundTrip:
    """bodies_to_text → parse_ic_text must reproduce the same bodies."""

    def test_solar_system_round_trip(self):
        original = solar_system_preset()
        text = bodies_to_text(original)
        parsed = parse_ic_text(text)

        assert len(parsed) == len(original), (
            f"Round-trip length mismatch: {len(parsed)} != {len(original)}"
        )
        for orig, back in zip(original, parsed):
            assert orig.name == back.name, f"Name mismatch: {orig.name!r} != {back.name!r}"
            assert math.isclose(orig.mass_kg, back.mass_kg, rel_tol=1e-4), (
                f"{orig.name}: mass_kg round-trip error: {orig.mass_kg} vs {back.mass_kg}"
            )


# ---------------------------------------------------------------------------
# parse_ic_text edge cases
# ---------------------------------------------------------------------------

class TestParseIcTextEdgeCases:
    """Comment lines, blank lines, and short lines."""

    def test_comment_lines_skipped(self):
        text = "# this is a comment\n# another comment\n"
        bodies = parse_ic_text(text)
        assert bodies == [], f"Expected empty list, got {bodies}"

    def test_blank_lines_skipped(self):
        text = "\n   \n\t\n"
        bodies = parse_ic_text(text)
        assert bodies == [], f"Expected empty list from blank-only text, got {bodies}"

    def test_short_line_raises_value_error(self):
        """A line with fewer than 8 comma-separated fields must raise ValueError."""
        text = "Earth, 5.972e24, 1.0, 0.0, 0.0, 29.784, 6371"  # only 7 fields
        with pytest.raises(ValueError):
            parse_ic_text(text)


# ---------------------------------------------------------------------------
# to_dict() structure
# ---------------------------------------------------------------------------

class TestToDictStructure:
    """GravityParams.to_dict() must include all required keys with correct values."""

    def test_required_keys_present(self):
        d = GravityParams().to_dict()
        required = {
            "bodies", "G", "softeningM", "dtS", "substeps",
            "timeWarp", "canvasId", "trailEnabled", "mergeEnabled",
        }
        missing = required - d.keys()
        assert not missing, f"to_dict() missing keys: {missing}"

    def test_first_body_is_sun(self):
        d = GravityParams().to_dict()
        assert d["bodies"][0]["name"] == "Sun"

    def test_sun_mass_correct(self):
        d = GravityParams().to_dict()
        sun_mass = d["bodies"][0]["massKg"]
        assert math.isclose(sun_mass, 1.989e30, rel_tol=0.01), (
            f"Sun massKg in to_dict() = {sun_mass:.3e}, expected ~1.989e30"
        )
