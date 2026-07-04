import numpy as np
import pytest

from rocketpy.rocket.parachutes import HemisphericalParachute
from rocketpy.simulation.flight import Flight


def test_trigger_receives_u_dot():
    def derivative_func(_t, _y):
        return np.array([0, 0, 0, 1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0, 0])

    recorded = {}

    def user_trigger(_p, _h, _y, u_dot):
        recorded["u_dot"] = np.array(u_dot)
        return True

    parachute = HemisphericalParachute(
        name="test",
        cd_s=1.0,
        trigger=user_trigger,
        sampling_rate=100,
    )

    dummy = type("D", (), {})()

    res = Flight._evaluate_parachute_trigger(
        dummy,
        parachute,
        pressure=0.0,
        height=10.0,
        y=np.zeros(13),
        sensors=[],
        derivative_func=derivative_func,
        t=0.0,
    )

    assert res is True
    assert "u_dot" in recorded
    assert np.allclose(recorded["u_dot"][3:6], np.array([1.0, 2.0, 3.0]))


def test_trigger_with_u_dot_only():
    """Test trigger that only expects u_dot (no sensors)."""

    def derivative_func(_t, _y):
        return np.array([0, 0, 0, -1.0, -2.0, -3.0, 0, 0, 0, 0, 0, 0, 0])

    recorded = {}

    def user_trigger(_p, _h, _y, u_dot):
        recorded["u_dot"] = np.array(u_dot)
        return False

    parachute = HemisphericalParachute(
        name="test_u_dot_only",
        cd_s=1.0,
        trigger=user_trigger,
        sampling_rate=100,
    )

    dummy = type("D", (), {})()

    res = Flight._evaluate_parachute_trigger(
        dummy,
        parachute,
        pressure=0.0,
        height=5.0,
        y=np.zeros(13),
        sensors=[],
        derivative_func=derivative_func,
        t=1.234,
    )

    assert res is False
    assert "u_dot" in recorded
    assert np.allclose(recorded["u_dot"][3:6], np.array([-1.0, -2.0, -3.0]))


def test_basic_trigger_does_not_compute_u_dot():
    def derivative_func(_t, _y):
        raise RuntimeError("derivative should not be called for legacy triggers")

    called = {}

    def basic_trigger(_p, _h, _y):
        called["ok"] = True
        return True

    parachute = HemisphericalParachute(
        name="basic",
        cd_s=1.0,
        trigger=basic_trigger,
        sampling_rate=100,
    )

    dummy = type("D", (), {})()

    res = Flight._evaluate_parachute_trigger(
        dummy,
        parachute,
        pressure=0.0,
        height=0.0,
        y=np.zeros(13),
        sensors=[],
        derivative_func=derivative_func,
        t=0.0,
    )

    assert res is True
    assert called.get("ok", False) is True


def test_five_arg_trigger_with_descriptive_names_receives_u_dot():
    """Regression: a documented 5-arg trigger (p, h, y, sensors, u_dot) with
    descriptive parameter names must still receive a valid (non-None) u_dot.
    u_dot delivery used to be gated on the parameter NAME, so such triggers
    silently received u_dot=None and crashed."""

    def derivative_func(_t, _y):
        return np.array([0, 0, 0, 4.0, 5.0, 6.0, 0, 0, 0, 0, 0, 0, 0])

    recorded = {}

    def user_trigger(pressure, altitude, state, sensor_list, state_derivative):
        recorded["u_dot"] = (
            None if state_derivative is None else np.array(state_derivative)
        )
        return True

    parachute = HemisphericalParachute(
        name="five_arg", cd_s=1.0, trigger=user_trigger, sampling_rate=100
    )
    dummy = type("D", (), {})()

    res = Flight._evaluate_parachute_trigger(
        dummy,
        parachute,
        pressure=0.0,
        height=10.0,
        y=np.zeros(13),
        sensors=[],
        derivative_func=derivative_func,
        t=0.0,
    )

    assert res is True
    assert recorded["u_dot"] is not None
    assert np.allclose(recorded["u_dot"][3:6], np.array([4.0, 5.0, 6.0]))


def test_four_arg_acceleration_trigger_by_name_receives_u_dot():
    """A 4-arg trigger whose 4th parameter is named like an acceleration must
    receive u_dot (name-based disambiguation is retained only for the ambiguous
    4-arg case)."""

    def derivative_func(_t, _y):
        return np.array([0, 0, 0, -9.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    recorded = {}

    def user_trigger(_p, _h, _y, acceleration):
        recorded["val"] = None if acceleration is None else np.array(acceleration)
        return True

    parachute = HemisphericalParachute(
        name="four_arg_acc", cd_s=1.0, trigger=user_trigger, sampling_rate=100
    )
    dummy = type("D", (), {})()

    res = Flight._evaluate_parachute_trigger(
        dummy,
        parachute,
        pressure=0.0,
        height=1.0,
        y=np.zeros(13),
        sensors=["S"],
        derivative_func=derivative_func,
        t=0.0,
    )

    assert res is True
    assert recorded["val"] is not None
    assert recorded["val"][3] == -9.0


def test_four_arg_sensors_trigger_receives_sensors():
    """A 4-arg trigger whose 4th parameter is named 'sensors' must receive the
    sensors list and must NOT trigger u_dot computation."""

    def derivative_func(_t, _y):
        raise RuntimeError("u_dot should not be computed for a sensors trigger")

    recorded = {}

    def user_trigger(_p, _h, _y, sensors):
        recorded["sensors"] = sensors
        return False

    parachute = HemisphericalParachute(
        name="four_arg_sensors", cd_s=1.0, trigger=user_trigger, sampling_rate=100
    )
    dummy = type("D", (), {})()

    res = Flight._evaluate_parachute_trigger(
        dummy,
        parachute,
        pressure=0.0,
        height=1.0,
        y=np.zeros(13),
        sensors=["A", "B"],
        derivative_func=derivative_func,
        t=0.0,
    )

    assert res is False
    assert recorded["sensors"] == ["A", "B"]


def test_invalid_arity_trigger_raises_at_construction():
    """A trigger with fewer than 3 parameters must fail fast at construction
    rather than mid-simulation."""

    def bad_trigger(_p, _h):
        return True

    with pytest.raises(TypeError, match="unsupported signature"):
        HemisphericalParachute(
            name="bad", cd_s=1.0, trigger=bad_trigger, sampling_rate=100
        )
