import pytest

from rocketpy.rocket.actuator.roll import RollControl
from rocketpy.rocket.rocket import Rocket


def test_roll_torque_actuator_dynamics_filters_command():
    roll_control = RollControl(
        sampling_rate=10,
        max_roll_torque=100,
        torque_rate_limit=1000,
        roll_torque=0,
        actuator_tau=0.4,
    )

    roll_control.roll_torque = 10

    assert roll_control.roll_torque == pytest.approx(2)
    assert roll_control.roll_torque_prev == pytest.approx(10)
    assert roll_control.roll_torque_filtered == pytest.approx(2)

    roll_control.roll_torque = 10

    assert roll_control.roll_torque == pytest.approx(3.6)


def test_roll_torque_without_actuator_dynamics_is_pass_through():
    roll_control = RollControl(
        sampling_rate=10,
        max_roll_torque=100,
        torque_rate_limit=1000,
        roll_torque=0,
    )

    roll_control.roll_torque = 10

    assert roll_control.roll_torque == pytest.approx(10)


def test_roll_control_reset_restores_filtered_state():
    roll_control = RollControl(
        sampling_rate=10,
        max_roll_torque=100,
        torque_rate_limit=1000,
        roll_torque=5,
        actuator_tau=0.4,
    )
    roll_control.roll_torque = 25

    roll_control._reset()

    assert roll_control.roll_torque == pytest.approx(5)
    assert roll_control.roll_torque_prev == pytest.approx(5)
    assert roll_control.roll_torque_filtered == pytest.approx(5)


def test_roll_control_serializes_actuator_tau():
    roll_control = RollControl(
        sampling_rate=20,
        max_roll_torque=50,
        torque_rate_limit=200,
        clamp=False,
        roll_torque=3,
        actuator_tau=0.2,
        name="Test Roll Control",
    )

    data = roll_control.to_dict()
    restored = RollControl.from_dict(data)

    assert data["actuator_tau"] == pytest.approx(0.2)
    assert restored.actuator_tau == pytest.approx(0.2)
    assert restored.roll_torque == pytest.approx(3)


def test_rocket_add_roll_control_forwards_actuator_tau():
    def controller_function(*_):
        return None

    rocket = Rocket(
        radius=0.1,
        mass=10,
        inertia=(1, 1, 1),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )

    roll_control = rocket.add_roll_control(
        max_roll_torque=20,
        torque_rate_limit=100,
        controller_function=controller_function,
        sampling_rate=50,
        actuator_tau=0.3,
    )

    assert roll_control.actuator_tau == pytest.approx(0.3)
