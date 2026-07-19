from rocketpy.simulation.flight import Flight
from rocketpy.stochastic import StochasticFlight


def test_stochastic_flight_create_object(stochastic_flight):
    obj = stochastic_flight.create_object()
    assert isinstance(obj, Flight)


def test_stochastic_flight_default_attributes(stochastic_flight, flight_calisto_robust):
    obj = stochastic_flight.create_object()
    assert flight_calisto_robust.max_time_step == obj.max_time_step
    assert flight_calisto_robust.min_time_step == obj.min_time_step
    assert flight_calisto_robust.rtol == obj.rtol
    assert flight_calisto_robust.atol == obj.atol
    assert flight_calisto_robust.name == obj.name
    assert flight_calisto_robust.equations_of_motion == obj.equations_of_motion
    assert flight_calisto_robust.ode_solver == obj.ode_solver
    assert flight_calisto_robust.simulation_mode == obj.simulation_mode


def test_stochastic_flight_optional_attributes(flight_calisto_robust):
    stochastic_flight = StochasticFlight(
        flight=flight_calisto_robust,
        inclination=(84.7, 1),
        heading=(53, 2),
        terminate_on_apogee=True,
        time_overshoot=True,
        max_time=987.6,
    )
    obj = stochastic_flight.create_object()
    assert obj.terminate_on_apogee is True
    assert obj.time_overshoot is True
    assert obj.max_time == 987.6
