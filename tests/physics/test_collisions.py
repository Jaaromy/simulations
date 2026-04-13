"""
Tests for CollisionParams and build_collision_html.

No Python physics logic exists — Matter.js runs the simulation in the browser.
So tests cover:
  1. Parameter validation — reject invalid configs before they reach the JS engine.
  2. HTML builder output — correct structure, dimensions, and config injection.
"""

import json

import pytest

from simulations.physics.collisions.params import CollisionParams
from simulations.physics.collisions.widget import build_collision_html

MATTERJS_MARKER = "Matter"  # present in both the inlined lib and config key checks


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_default_params_valid() -> None:
    """Default CollisionParams must pass validation without error."""
    CollisionParams().validate()


def test_n_balls_negative_raises() -> None:
    with pytest.raises(ValueError, match="n_balls"):
        CollisionParams(n_balls=-1).validate()


def test_n_balls_too_many_raises() -> None:
    with pytest.raises(ValueError, match="n_balls"):
        CollisionParams(n_balls=201).validate()


def test_n_balls_zero_valid() -> None:
    """Zero balls is a valid edge case — empty simulation."""
    CollisionParams(n_balls=0).validate()


def test_n_balls_boundary_valid() -> None:
    CollisionParams(n_balls=200).validate()


def test_friction_below_zero_raises() -> None:
    with pytest.raises(ValueError, match="friction"):
        CollisionParams(friction=-0.01).validate()


def test_friction_above_one_raises() -> None:
    with pytest.raises(ValueError, match="friction"):
        CollisionParams(friction=1.001).validate()


def test_friction_boundaries_valid() -> None:
    CollisionParams(friction=0.0).validate()
    CollisionParams(friction=1.0).validate()


def test_restitution_below_zero_raises() -> None:
    with pytest.raises(ValueError, match="restitution"):
        CollisionParams(restitution=-0.1).validate()


def test_restitution_above_one_raises() -> None:
    with pytest.raises(ValueError, match="restitution"):
        CollisionParams(restitution=1.5).validate()


def test_restitution_boundaries_valid() -> None:
    CollisionParams(restitution=0.0).validate()
    CollisionParams(restitution=1.0).validate()


def test_mass_min_non_positive_raises() -> None:
    with pytest.raises(ValueError, match="mass_min"):
        CollisionParams(mass_min=0.0).validate()
    with pytest.raises(ValueError, match="mass_min"):
        CollisionParams(mass_min=-1.0).validate()


def test_mass_max_non_positive_raises() -> None:
    with pytest.raises(ValueError, match="mass_max"):
        CollisionParams(mass_max=0.0).validate()


def test_mass_min_greater_than_mass_max_raises() -> None:
    with pytest.raises(ValueError, match="mass_min"):
        CollisionParams(mass_min=6.0, mass_max=5.0).validate()


def test_mass_min_equal_mass_max_valid() -> None:
    """All balls same mass is valid."""
    CollisionParams(mass_min=3.0, mass_max=3.0).validate()


def test_speed_negative_raises() -> None:
    with pytest.raises(ValueError, match="speed"):
        CollisionParams(speed=-1.0).validate()


def test_speed_zero_valid() -> None:
    """Zero initial speed → balls start at rest."""
    CollisionParams(speed=0.0).validate()


def test_gravity_negative_raises() -> None:
    with pytest.raises(ValueError, match="gravity_y"):
        CollisionParams(gravity_y=-0.1).validate()


def test_gravity_zero_valid() -> None:
    """Zero gravity → billiards / space mode."""
    CollisionParams(gravity_y=0.0).validate()


# ---------------------------------------------------------------------------
# HTML builder — structure
# ---------------------------------------------------------------------------

@pytest.fixture
def default_html() -> str:
    return build_collision_html(CollisionParams())


def test_html_nonempty(default_html: str) -> None:
    assert len(default_html) > 100


def test_html_contains_canvas(default_html: str) -> None:
    # Content is HTML-escaped inside the srcdoc attribute value
    assert "&lt;canvas" in default_html


def test_html_contains_matterjs(default_html: str) -> None:
    # Matter.js is inlined — check the global name is present in the srcdoc
    assert MATTERJS_MARKER in default_html


def test_html_contains_config_json(default_html: str) -> None:
    """Inline JS must include a 'config' object with serialised params."""
    assert "config" in default_html


# ---------------------------------------------------------------------------
# HTML builder — parameter injection
# ---------------------------------------------------------------------------

def test_canvas_dimensions_in_html() -> None:
    params = CollisionParams(canvas_width=1024, canvas_height=768)
    html = build_collision_html(params)
    assert "1024" in html
    assert "768" in html


def test_config_json_values_in_html() -> None:
    """The params dict must be present in the srcdoc content (HTML-escaped)."""
    import html as html_module
    params = CollisionParams(
        n_balls=7,
        friction=0.33,
        restitution=0.55,
        gravity_y=2.0,
    )
    output = build_collision_html(params)
    # The srcdoc attribute contains the inner HTML document, HTML-escaped.
    # Unescape it to recover the raw document and check keys/values there.
    inner = html_module.unescape(output)
    config_dict = params.to_dict()
    for key, value in config_dict.items():
        assert f'"{key}"' in inner, f"Key '{key}' missing from srcdoc"
        assert str(value).lower() in inner.lower(), f"Value for '{key}' missing from srcdoc"


def test_zero_balls_produces_valid_html() -> None:
    """Edge case: n_balls=0 must not raise and must return non-empty HTML."""
    params = CollisionParams(n_balls=0)
    html = build_collision_html(params)
    assert len(html) > 50
    assert "&lt;canvas" in html  # escaped inside srcdoc attribute


def test_velocity_vectors_flag_in_html() -> None:
    html_on = build_collision_html(CollisionParams(show_velocity_vectors=True))
    html_off = build_collision_html(CollisionParams(show_velocity_vectors=False))
    assert "true" in html_on
    assert "false" in html_off


def test_invalid_params_raises_before_building_html() -> None:
    """build_collision_html must call validate() and propagate errors."""
    with pytest.raises(ValueError):
        build_collision_html(CollisionParams(n_balls=-1))
